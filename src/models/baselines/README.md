# Active baseline wrappers

`tabpfn_model.py` is the TabPFN wrapper used by the UP5 target-only benchmark.
It imports the installed `tabpfn` package, not the old vendored helper code in the
model archive. The wrapper was moved here unchanged. This subpackage avoids adding
a top-level model file to the existing validation-sweep fingerprint.
