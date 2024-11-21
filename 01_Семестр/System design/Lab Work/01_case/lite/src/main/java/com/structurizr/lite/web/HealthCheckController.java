package com.structurizr.lite.web;

import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HealthCheckController extends AbstractController {

    @RequestMapping(value = "/health", method = RequestMethod.GET, produces = "text/plain")
    public String showHealthCheckPage(ModelMap model) {
        return "OK";
    }

}