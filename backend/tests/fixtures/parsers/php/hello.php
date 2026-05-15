<?php

namespace App;

use Other\Base;

class Greeter extends Base {
    public function greet($name) {
        $formatted = $this->formatName($name);
        echo $formatted;
    }

    private function formatName($name) {
        return strtoupper($name);
    }
}

function standaloneHelper() {
    $g = new Greeter();
    $g->greet("world");
}
