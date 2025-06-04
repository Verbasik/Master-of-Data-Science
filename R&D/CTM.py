# -*- coding: utf-8 -*-
"""
ctm_train.py  ─ Continuous-Thought-Machine с attention, синхронизацией,
two-point loss по всей строке + комбинированный LM-loss + cross-attention
в декодере + расширенная логгинг-статистика (token-acc, ppl, ECE, BLEU, ROUGE).
"""

import os, glob, pickle, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# ─── дополнительные импорты для ECE ───
from torchmetrics.classification import CalibrationError

smooth = SmoothingFunction().method3
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# ───────────────────────── Dataset & DataLoader ─────────────────────────

class EEGEpochSpectrogramDataset(Dataset):
    """EEG → лог-спектр ≤ 50 Гц + токены текста."""

    def __init__(
        self,
        pkl_paths: List[str],
        tokenizer,
        sample_rate: int = 500,
        n_fft: int = 256,
        hop_length: int = 128,
        max_len_tokens: int = 64,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.sr, self.n_fft, self.hop = sample_rate, n_fft, hop_length
        self.max_len = max_len_tokens

        # Загрузка всех .pkl
        self._records: List[Dict[str, Any]] = []
        for p in pkl_paths:
            with open(p, "rb") as f:
                self._records.extend(pickle.load(f))

        # Предвычисляем маску ≤50 Гц и окно Ханна
        freqs = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        self.freq_mask = (freqs <= 50.0)
        self.window    = torch.hann_window(self.n_fft)

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _to_tensor(x: Any) -> Tensor:
        return torch.as_tensor(x, dtype=torch.float32).contiguous()

    def _compute_spectrogram(self, sig: Tensor) -> Tensor:
        win  = self.window.to(sig.device)
        mask = self.freq_mask.to(sig.device)
        spec = torch.stft(
            sig, n_fft=self.n_fft, hop_length=self.hop,
            win_length=self.n_fft, window=win, return_complex=True
        )
        power = spec.abs().pow(2)[:, mask, :]
        return torch.log1p(power)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Dict]:
        rec = self._records[idx]
        signal = self._to_tensor(rec["input_features"])
        if signal.dim() == 3 and signal.size(0) == 1:
            signal = signal.squeeze(0)
        spec = self._compute_spectrogram(signal)

        # Преобразуем любое значение в строку, nan станет "nan"
        text = str(rec.get("text", ""))
        if text == "nan": text = "[MISSING]"  # при условии хуевой обработки
        ids = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return spec, ids, {"raw_text": text, "idx": idx}


def create_dataloader(
    pkl_dir: str,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    num_ticks: int,
    **kwargs,
) -> DataLoader:
    
    # # Собираем пути
    paths = glob.glob(os.path.join(pkl_dir, "**.pkl"))
    dataset = EEGEpochSpectrogramDataset(paths, tokenizer, **kwargs)

    def _collate(batch):
        specs, toks, metas = zip(*batch)
        max_T = max(s.size(-1) for s in specs)
        rem   = max_T % num_ticks
        if rem:
            max_T += (num_ticks - rem)
        pad_s = [F.pad(s, (0, max_T - s.size(-1))) for s in specs]
        return torch.stack(pad_s), torch.stack(toks), metas

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )

# ───────────────────────── Model blocks ────────────────────────────────

class DepthwiseConv2d(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        p = k // 2
        self.depth = nn.Conv2d(c_in, c_in, k, padding=p, groups=c_in)
        self.point = nn.Conv2d(c_in, c_out, 1)
        self.bn    = nn.BatchNorm2d(c_out)
        self.act   = nn.GELU()
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.point(self.depth(x))))


class EEGEncoder(nn.Module):
    def __init__(self, c_in: int, hidden: int = 128, layers: int = 3):
        super().__init__()
        seq, c = [], c_in
        for _ in range(layers):
            seq.append(DepthwiseConv2d(c, hidden)); c = hidden
        self.cnn  = nn.Sequential(*seq)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
    def forward(self, x: Tensor) -> Tensor:
        h = self.cnn(x)                  
        h = self.pool(h).squeeze(2)      
        return h.permute(0, 2, 1)        


class SynapseModule(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.up   = nn.Sequential(
            nn.Linear(2*d, 2*d), nn.GELU(),
            nn.Linear(2*d, d),   nn.GELU()
        )
        self.down = nn.Sequential(
            nn.Linear(d, 2*d),   nn.GELU(),
            nn.Linear(2*d, d)
        )
    def forward(self, z: Tensor, o: Tensor) -> Tensor:
        return self.down(self.up(torch.cat([z, o], -1)))


class CTMBlock(nn.Module):
    def __init__(
        self,
        D: int,
        H: int,
        M: int,
        k_out: int = 8,
        k_act: int = 8
    ):
        super().__init__()
        self.D, self.M = D, M
        self.syn = SynapseModule(D)
        self.nlm = nn.Sequential(
            nn.Linear(D*M, H), nn.GELU(),
            nn.Linear(H, D),
        )
        k_out = min(k_out, D)
        k_act = min(k_act, D)
        self.register_buffer("I_out",    torch.arange(k_out))
        self.register_buffer("I_action", torch.arange(D-k_act, D))
        self.W_out = nn.Linear(D*k_out, H)
        self.W_in  = nn.Sequential(
            nn.Linear(D*k_act, D),
            nn.LayerNorm(D),
        )

    @staticmethod
    def _sync(z_hist: Tensor) -> Tensor:
        z_norm = F.normalize(z_hist, dim=2)
        return torch.matmul(z_norm, z_norm.transpose(1,2)) / z_norm.size(2)

    def forward(self, z_hist: Tensor, o_t: Tensor):
        B, D, M = z_hist.shape
        z_last  = z_hist[:, :, -1]
        a_t     = self.syn(z_last, o_t)
        z_new   = self.nlm(z_hist.flatten(1))
        z_next  = z_last + a_t + z_new

        S      = self._sync(z_hist)
        # лог S-норм/var
        if self.training and hasattr(self, "logger"):
            sync_norm = S.pow(2).sum((-2,-1)).sqrt().mean()
            sync_var  = S.var((-2,-1)).mean()
            self.logger.add_scalar("train/sync_norm", sync_norm.item(), self.global_step)
            self.logger.add_scalar("train/sync_var",  sync_var.item(),  self.global_step)

        s_out  = S[:, self.I_out, :].reshape(B, -1)
        s_act  = S[:, self.I_action, :].reshape(B, -1)

        y_t_hidden = self.W_out(s_out)
        q_next     = self.W_in(s_act)
        return z_next, q_next, y_t_hidden


# ───────────────────── Full CTM + cross-attn LM ────────────────────────

@dataclass
class CTMConf:
    channels:  int
    hidden:    int = 128
    neurons:   int = 64
    ticks:     int = 8
    window_M:  int = 8
    vocab:     int = 30000
    max_len:   int = 64


class CTMModel(nn.Module):
    def __init__(self, cfg: CTMConf):
        super().__init__()
        self.cfg = cfg
        D, H = cfg.neurons, cfg.hidden
        C    = cfg.channels

        self.obs_encoder = EEGEncoder(C, H)
        self.key_proj    = nn.Linear(H, D, bias=False)
        self.value_proj  = nn.Linear(H, D, bias=False)
        self.ctm_block   = CTMBlock(D, H, cfg.window_M)

        self.token_emb   = nn.Embedding(cfg.vocab, H)
        self.pos_emb     = nn.Parameter(torch.zeros(cfg.max_len, H))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.decoder     = nn.GRU(H, H, batch_first=True)
        self.to_logits   = nn.Linear(H, cfg.vocab)
        self.dec_k_proj  = nn.Linear(H, H, bias=False)
        self.dec_v_proj  = nn.Linear(H, H, bias=False)

        # logger placeholders
        self.logger = None
        self.global_step = 0

    def forward(
        self,
        spec:   Tensor,
        tgt_ids:Optional[Tensor] = None
    ) -> Tuple[List[Tensor], Optional[Tensor], Optional[Tensor]]:
        B, C, F, Tspec = spec.shape
        T              = self.cfg.ticks
        assert Tspec % T == 0, f"Tspec={Tspec} must divide ticks={T}"
        step = Tspec // T

        D, M = self.cfg.neurons, self.cfg.window_M
        z_hist = spec.new_zeros(B, D, M)
        q_prev = spec.new_zeros(B, D)

        y_seq: List[Tensor] = []
        logits_seq: List[Tensor] = []

        for t in range(T):
            frame = spec[..., t*step:(t+1)*step]
            enc   = self.obs_encoder(frame)           # (B,step,H)

            K_enc = self.key_proj(enc)
            V_enc = self.value_proj(enc)
            att   = torch.softmax(torch.einsum('bd,btd->bt', q_prev, K_enc), dim=-1)

            # attention entropy (энкодер)
            if self.training and self.logger:
                ent_enc = -(att * (att + 1e-12).log()).sum(-1).mean()
                self.logger.add_scalar("train/att_ent_enc", ent_enc.item(), self.global_step)

            o_t   = torch.einsum('bt,btd->bd', att, V_enc)

            z_next, q_prev, y_t = self.ctm_block(z_hist, o_t)
            logits_t = self.to_logits(y_t)
            logits_seq.append(logits_t)
            y_seq.append(y_t)

            z_hist = torch.cat([z_hist[:, :, 1:], z_next.unsqueeze(-1)], dim=-1)

        lm_logits = None
        if tgt_ids is not None:
            tok_emb = self.token_emb(tgt_ids) + self.pos_emb[: tgt_ids.size(1)]
            out, _  = self.decoder(tok_emb)            # (B,L,H)

            # cross-attn в декодере
            mem      = torch.stack(y_seq, dim=1)       # (B,T,H)
            K_dec    = self.dec_k_proj(mem)
            V_dec    = self.dec_v_proj(mem)
            att_dec  = torch.softmax(torch.einsum('blh,bth->blt', out, K_dec), dim=-1)

            # attention entropy (декодер)
            if self.training and self.logger:
                ent_dec = -(att_dec * (att_dec + 1e-12).log()).sum(-1).mean()
                self.logger.add_scalar("train/att_ent_dec", ent_dec.item(), self.global_step)

            ctx_dec  = torch.einsum('blt,bth->blh', att_dec, V_dec)
            out      = out + ctx_dec
            lm_logits= self.to_logits(out)

        return logits_seq, lm_logits, tgt_ids


# ───────────────────── two-point loss, metrics & training ───────────────────────

PAD_CACHE: Dict[str,int] = {}

def get_pad_id(tok: AutoTokenizer) -> int:
    if tok.name_or_path not in PAD_CACHE:
        pid = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        PAD_CACHE[tok.name_or_path] = pid if pid is not None else -100
    return PAD_CACHE[tok.name_or_path]


def tp_loss_and_stats(
    logits_seq: List[Tensor],
    targets:     Tensor,
    pad:         int,
    eps_ce:      float = 1.0
) -> Tuple[Tensor, Dict[str, float]]:
    B, L = targets.size()
    mask_flat = targets.ne(pad).reshape(-1)
    tgt_flat  = targets.reshape(-1)[mask_flat]

    ce_by_tau, conf_by_tau = [], []
    for lg in logits_seq:
        all_logits = lg.unsqueeze(1).expand(-1, L, -1).reshape(-1, lg.size(-1))
        logits_n   = all_logits[mask_flat]
        ce_n       = F.cross_entropy(logits_n, tgt_flat, reduction="none")
        p          = logits_n.softmax(-1)
        conf_n     = -(p * p.log()).sum(-1)
        ce_by_tau.append(ce_n); conf_by_tau.append(conf_n)

    ce_all  = torch.stack(ce_by_tau)    # (T, N)
    conf_all= torch.stack(conf_by_tau)  # (T, N)

    good      = ce_all < eps_ce
    csum      = good.float().cumsum(0)
    first_good= (csum==1) & good
    has_good  = good.any(0)
    t1        = torch.where(has_good, first_good.float().argmax(0), ce_all.argmin(0))
    t2        = conf_all.argmax(0)

    idx   = torch.arange(ce_all.size(1))
    ce_t1 = ce_all[t1, idx]; ce_t2 = ce_all[t2, idx]
    loss  = (2*ce_t1 + ce_t2).mean()

    stats = {
        "CE_t1":      ce_t1.mean().item(),
        "CE_t2":      ce_t2.mean().item(),
        "mean_t1":    t1.float().mean().item(),
        "tick_saved": 1.0 - t1.float().mean().item()/len(logits_seq),
    }
    return loss, stats


def token_accuracy(logits: Tensor, targets: Tensor, pad: int) -> float:
    pred = logits.argmax(-1)  # handles (B,V) or (B,L,V)
    valid= targets.ne(pad)
    correct = pred.eq(targets) & valid
    return correct.sum().item() / valid.sum().item()


@dataclass
class TrainArgs:
    data_root:      str   = "/Users/cyberrunner/Documents/Code/me/Paper/Study/output/sub-01/train"
    logdir:         str   = "./runs/ctm"
    ckpt_dir:       str   = "./checkpoints"
    epochs:         int   = 20
    batch_size:     int   = 8
    lr:             float = 2e-4
    weight_decay:   float = 1e-2
    max_grad_norm:  float = 1.0
    n_fft:          int   = 256
    hop_length:     int   = 128
    num_ticks:      int   = 8
    window_M:       int   = 8
    hidden:         int   = 128
    lm_loss_weight: float = 0.3
    eps_ce:         float = 1.0


def greedy_decode(model, spec, max_len, bos_id, eos_id):
    ys = [bos_id]
    for _ in range(max_len):
        inp = torch.tensor(ys, device=spec.device).unsqueeze(0)
        logits_seq, lm_logits, _ = model(spec, inp)
        next_id = lm_logits[:, -1].argmax(-1).item()
        ys.append(next_id)
        if next_id == eos_id:
            break
    return ys


def load_checkpoint(checkpoint_path: str, model: CTMModel, 
                   optim: Optional[AdamW] = None, 
                   sched: Optional[torch.optim.lr_scheduler.OneCycleLR] = None,
                   device: str = 'cuda'):
    """Загрузка checkpoint для продолжения обучения или инференса"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optim is not None and 'optimizer_state_dict' in checkpoint:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if sched is not None and 'scheduler_state_dict' in checkpoint:
        sched.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    global_step = checkpoint.get('global_step', 0)
    
    print(f"✓ Checkpoint loaded from epoch {checkpoint.get('epoch', 0)}")
    return start_epoch, global_step


def train(args: TrainArgs = TrainArgs(), resume_from: Optional[str] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok    = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    pad    = get_pad_id(tok)

    # ECE-метрика
    ece_metric = CalibrationError(
        task="multiclass",
        n_bins=15,
        num_classes=len(tok)  # или cfg.vocab
    ).to(device)

    # DataLoaders
    tr_loader = create_dataloader(
        args.data_root, tok,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        num_ticks=args.num_ticks,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )

    print(f"Dataset tr_loader has {len(tr_loader.dataset)} files / "
        f"{len(tr_loader)} batches", flush=True)

    vl_loader = create_dataloader(
        args.data_root, tok,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        num_ticks=args.num_ticks,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )

    print(f"Dataset vl_loader has {len(vl_loader.dataset)} files / "
        f"{len(vl_loader)} batches", flush=True)

    spec0, _, _ = next(iter(tr_loader))
    conf = CTMConf(
        channels=spec0.shape[1],
        hidden  = args.hidden,
        neurons = spec0.shape[1],
        ticks   = args.num_ticks,
        window_M= args.window_M,
        vocab   = len(tok),
        max_len = tok.model_max_length
    )
    model = CTMModel(conf).to(device)

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr,
        steps_per_epoch=len(tr_loader),
        epochs=args.epochs
    )
    writer = SummaryWriter(args.logdir)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # передаём logger в модель
    model.logger = writer

    # Инициализация переменных для отслеживания лучшей модели
    best_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 0
    global_step = 0

    # Загрузка checkpoint если указан
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        start_epoch, global_step = load_checkpoint(
            resume_from, model, optim, sched, device
        )
        model.global_step = global_step
    else:
        model.global_step = 0

    for ep in range(start_epoch, args.epochs):
        # ── TRAIN ──
        model.train()
        total_train_lm_ce = 0.0
        total_train_steps = 0

        for specs, ids, _ in tr_loader:
            specs, ids = specs.to(device), ids.to(device)

            logits_seq, lm_logits, _ = model(specs, ids)
            tp_loss, stats       = tp_loss_and_stats(logits_seq, ids, pad, eps_ce=args.eps_ce)

            # LM CE по всей sequence
            V     = lm_logits.size(-1)
            lm_ce = F.cross_entropy(
                lm_logits.reshape(-1, V),
                ids.reshape(-1),
                ignore_index=pad
            )

            loss = tp_loss + args.lm_loss_weight * lm_ce

            optim.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            sched.step()

            # логгирование per-step
            writer.add_scalar("train/loss_iter", loss.item(), global_step)
            writer.add_scalar("train/lm_ce", lm_ce.item(), global_step)
            for k,v in stats.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], global_step)
            nan_ratio = torch.isnan(loss).float().mean().item()
            writer.add_scalar("debug/nan_ratio", nan_ratio, global_step)

            # сохраняем для epoch-статистик
            total_train_lm_ce += lm_ce.item()
            total_train_steps += 1

            # обновляем шаг
            model.global_step = global_step
            global_step += 1

        # за эпоху
        train_lm_ce_epoch = total_train_lm_ce / total_train_steps
        writer.add_scalar("train/lm_ce_epoch", train_lm_ce_epoch, ep)

        # ── VALIDATION ──
        model.eval()
        total_val_tp       = 0.0
        total_val_lm_ce    = 0.0
        total_val_tok      = 0
        val_tok_acc_accum  = 0.0

        ece_metric.reset()

        # периодическое генерирование для BLEU/ROUGE
        if ep % 2 == 0:
            refs, hyps = [], []
            samples = 0

        with torch.no_grad():
            for specs, ids, _ in vl_loader:
                specs, ids = specs.to(device), ids.to(device)

                logits_seq, lm_logits, _ = model(specs, ids)
                val_tp, _ = tp_loss_and_stats(logits_seq, ids, pad, eps_ce=args.eps_ce)
                total_val_tp += val_tp.item()

                # LM CE суммарно
                V = lm_logits.size(-1)
                ce_sum = F.cross_entropy(
                    lm_logits.reshape(-1, V),
                    ids.reshape(-1),
                    ignore_index=pad,
                    reduction="sum"
                )
                total_val_lm_ce += ce_sum.item()
                ntoks = ids.ne(pad).sum().item()
                total_val_tok += ntoks

                # token-accuracy
                pred = lm_logits.argmax(-1)
                val_tok_acc_accum += (pred.eq(ids)&ids.ne(pad)).sum().item()

                # ECE
                logits_probs = lm_logits.reshape(-1, V).softmax(-1)
                tgt_flat     = ids.reshape(-1)
                ece_metric.update(logits_probs, tgt_flat)

                # BLEU/ROUGE samples
                if ep % 2 == 0 and samples < 100:
                    for b in range(specs.size(0)):
                        refs.append([tok.decode(ids[b].cpu().tolist(), skip_special_tokens=True).split()])
                        hyp_ids = greedy_decode(
                            model, specs[b:b+1], max_len=ids.size(1),
                            bos_id=tok.bos_token_id or tok.cls_token_id,
                            eos_id=tok.eos_token_id or tok.sep_token_id
                        )
                        hyps.append(tok.decode(hyp_ids, skip_special_tokens=True).split())
                        samples += 1
                        if samples >= 100: break

        # epoch metrics
        val_tp_epoch   = total_val_tp / len(vl_loader)
        val_lm_ce_epoch= total_val_lm_ce / total_val_tok
        val_ppl        = math.exp(val_lm_ce_epoch)
        val_tok_acc    = val_tok_acc_accum / total_val_tok
        val_ece        = ece_metric.compute().item()

        writer.add_scalar("val/tp_loss",       val_tp_epoch, ep)
        writer.add_scalar("val/lm_ce_epoch",   val_lm_ce_epoch, ep)
        writer.add_scalar("val/token_acc",     val_tok_acc, ep)
        writer.add_scalar("val/ppl",           val_ppl, ep)
        writer.add_scalar("val/ece",           val_ece, ep)

        if ep % 2 == 0:
            bleu4 = corpus_bleu(refs, hyps, smoothing_function=smooth)
            rouge_L = rouge.score(
                " ".join(refs[0][0]), " ".join(hyps[0])
            )["rougeL"].fmeasure
            writer.add_scalar("val/bleu4", bleu4, ep)
            writer.add_scalar("val/rouge_L", rouge_L, ep)

        print(
            f"Ep{ep} TRAIN_LM_CE={train_lm_ce_epoch:.3f} "
            f"VAL_TP={val_tp_epoch:.3f} PPL={val_ppl:.1f} "
            f"TOK_ACC={val_tok_acc*100:.1f}% ECE={val_ece:.3f}"
        )

        # ── CHECKPOINT SAVING ──
        
        # Сохранение checkpoint каждые 5 эпох
        if (ep + 1) % 5 == 0:
            checkpoint = {
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'loss': val_tp_epoch,
                'val_lm_ce': val_lm_ce_epoch,
                'val_ppl': val_ppl,
                'val_token_acc': val_tok_acc,
                'config': conf,
                'global_step': global_step,
                'args': args
            }
            checkpoint_path = os.path.join(args.ckpt_dir, f'checkpoint_epoch_{ep+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Checkpoint saved at epoch {ep+1}: {checkpoint_path}")
        
        # Сохранение лучшей модели по комбинированной метрике
        current_val_loss = val_tp_epoch + val_lm_ce_epoch
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = ep
            
            best_checkpoint = {
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'loss': current_val_loss,
                'val_tp_loss': val_tp_epoch,
                'val_lm_ce': val_lm_ce_epoch,
                'val_ppl': val_ppl,
                'val_token_acc': val_tok_acc,
                'val_ece': val_ece,
                'config': conf,
                'global_step': global_step,
                'args': args,
                'is_best': True
            }
            best_model_path = os.path.join(args.ckpt_dir, 'best_model.pth')
            torch.save(best_checkpoint, best_model_path)
            print(f"✓ New best model saved! Epoch {ep+1}, Combined Loss: {current_val_loss:.4f}")
        
        # Сохранение последней модели (каждую эпоху)
        last_checkpoint = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'loss': current_val_loss,
            'val_tp_loss': val_tp_epoch,
            'val_lm_ce': val_lm_ce_epoch,
            'val_ppl': val_ppl,
            'val_token_acc': val_tok_acc,
            'config': conf,
            'global_step': global_step,
            'args': args
        }
        last_model_path = os.path.join(args.ckpt_dir, 'last_model.pth')
        torch.save(last_checkpoint, last_model_path)

    # Сохранение финальной модели
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'config': conf,
        'global_step': global_step,
        'args': args,
        'training_complete': True,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    final_model_path = os.path.join(args.ckpt_dir, 'final_model.pth')
    torch.save(final_checkpoint, final_model_path)
    
    # Создание summary файла с информацией о тренировке
    training_summary = {
        'total_epochs': args.epochs,
        'best_epoch': best_epoch + 1,
        'best_val_loss': best_val_loss,
        'final_global_step': global_step,
        'model_config': conf.__dict__,
        'training_args': args.__dict__,
        'checkpoint_files': {
            'best_model': 'best_model.pth',
            'last_model': 'last_model.pth', 
            'final_model': 'final_model.pth'
        }
    }
    
    summary_path = os.path.join(args.ckpt_dir, 'training_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== TRAINING SUMMARY ===\n\n")
        f.write(f"Training completed successfully!\n")
        f.write(f"Total epochs: {args.epochs}\n")
        f.write(f"Best epoch: {best_epoch + 1}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Final global step: {global_step}\n\n")
        f.write("Available checkpoints:\n")
        f.write(f"- best_model.pth (epoch {best_epoch + 1})\n")
        f.write(f"- last_model.pth (epoch {args.epochs})\n")
        f.write(f"- final_model.pth (training complete)\n\n")
        f.write("Model configuration:\n")
        for k, v in conf.__dict__.items():
            f.write(f"  {k}: {v}\n")

    writer.close()
    print(f"✓ Training complete! Best model at epoch {best_epoch + 1}")
    print(f"✓ Checkpoints saved in: {args.ckpt_dir}")
    print(f"✓ Training summary: {summary_path}")