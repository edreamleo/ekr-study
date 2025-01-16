//@+leo-ver=5-thin
//@+node:ekr.20240927151332.1: * @file src/main.rs
//@@language rust
// main.rs

// #![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

// For the f! macro.
#[macro_use]
extern crate fstrings;

mod beautifier;

fn main() {
    beautifier::entry();
}
//@-leo
