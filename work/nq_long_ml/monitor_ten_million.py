"""Local status writer; completes the authorized separate-period replay after discovery."""
from pathlib import Path
import json,subprocess,sys,time
R=Path('work/nq_long_ml/ten_million_entry_search')
while True:
    try:s=json.loads((R/'progress.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(30);continue
    text=f'''# NQ entry discovery: local run status

Status: **{s['status']}**

- Unique canonical definitions evaluated: **{s['definitions_tested']:,} / 10,000,000**
- Distinct passing in-sample signal streams saved: **{s['saved_unique_signals']:,}**
- Duplicate passing streams excluded: **{s['duplicate_is_streams']:,}**
- Below save cutoff / no qualifying events: **{s['failed_save_cutoff']:,}**
- Four local compute threads; checkpointed after every batch.

A saved signal exceeded 53% in-sample target hits. Saving is not proof of profitability or an institutional-grade edge. Candidates have individual JSON files inside ZIP archives, organized as passed validation, failed validation, or insufficient validation support. Rejected records are in the rejected folder. Ten million definitions are variations of documented quantitative families, not ten million independent research ideas.

Primary exit: 70% of past mean high-minus-open; one-hour forward duration capped at the cash-session close; no stops. Other move-size and percentile references are recorded separately. Unknown outcomes are conservative failures in raw hit rates and remain unresolved in execution summaries.

IS: October 2022–2024. Validation: 2025. Separate confirmation: 2026, replayed only after a supported profitable shortlist is frozen. The 2026 dates were previously used in other research and aggregate inspection; they are not an historically untouched blind holdout. Volume-roll and bar-start export provenance remain unverified. No live orders.
'''
    (R/'STATUS.md').write_text(text,encoding='utf-8')
    if s['status']=='discovery_complete_final_confirmation_pending':
        with (R/'confirmation.log').open('w') as stream:
            done=subprocess.run([sys.executable,'work/nq_long_ml/confirm_ten_million.py'],stdout=stream,stderr=subprocess.STDOUT)
        if done.returncode:
            s['status']='confirmation_error_preserved_results';(R/'progress.json').write_text(json.dumps(s,indent=2))
        continue
    if s['status']=='paused_by_stop_file' and not (R/'STOP_REQUESTED').exists():time.sleep(30);continue
    if s['status']!='running':break
    time.sleep(30)
