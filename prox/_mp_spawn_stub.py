"""
Dedicated reimport target for multiprocessing spawn workers - see
compare_segments() in segments.py for why this exists.

Windows (and macOS) multiprocessing can only spawn worker processes, not
fork. A spawned worker's bootstrap unconditionally reimports whatever file
sys.modules['__main__'] points to in the parent, via runpy, to rebuild a
consistent global environment. Under `streamlit run`, that's normally the
user's Streamlit script, full of top-level st.* calls that crash outside a
real Streamlit session. compare_segments() temporarily points __main__ at
this file instead for the duration of a parallel run.

This file must have zero side effects and zero imports of its own, since
runpy executes it as a bare script with no package context - a relative
import here would fail exactly the way it would in any other standalone
script.
"""
