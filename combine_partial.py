import os
import pickle
import shutil

partial_base = 'trained'

# find all partial_* directories
partial_dirs = [
    d for d in os.listdir(partial_base)
    if d.startswith('partial_') and os.path.isdir(os.path.join(partial_base, d))
]

print(f"Found {len(partial_dirs)} partial directories: {partial_dirs}")

for partial_dir_name in partial_dirs:
    partial_dir = os.path.join(partial_base, partial_dir_name)

    # parse model_name and sample_tag from folder name: partial_{model}_sample{tag}
    rest = partial_dir_name[len('partial_'):]               # e.g. "XGB_sample0_5"
    model_part, sample_part = rest.rsplit('_sample', 1)     # e.g. ("XGB", "0_5")
    model_name = model_part
    sample_tag = sample_part

    print(f"\nProcessing: model={model_name}, sample={sample_tag}")

    # collect all strategies from files present
    strategies = set()
    for fname in os.listdir(partial_dir):
        if fname.endswith('_model.pkl'):
            strategies.add(fname.replace('_model.pkl', ''))
        elif fname.endswith('_metrics.pkl'):
            strategies.add(fname.replace('_metrics.pkl', ''))

    # check both files exist for each strategy
    complete, incomplete = [], []
    for strategy in sorted(strategies):
        has_model   = os.path.exists(os.path.join(partial_dir, f'{strategy}_model.pkl'))
        has_metrics = os.path.exists(os.path.join(partial_dir, f'{strategy}_metrics.pkl'))
        if has_model and has_metrics:
            complete.append(strategy)
        else:
            incomplete.append(strategy)

    if incomplete:
        print(f"  WARNING: incomplete strategies (skipping): {incomplete}")

    print(f"  Combining {len(complete)} strategies: {complete}")

    trained_model = {}
    model_results = {}
    for strategy in complete:
        with open(os.path.join(partial_dir, f'{strategy}_model.pkl'),   'rb') as f:
            trained_model[strategy] = pickle.load(f)
        with open(os.path.join(partial_dir, f'{strategy}_metrics.pkl'), 'rb') as f:
            model_results[strategy] = pickle.load(f)

    os.makedirs('evaluation', exist_ok=True)

    out_model  = os.path.join(partial_base, f'{model_name}_sample{sample_tag}.pkl')
    out_eval   = os.path.join('evaluation', f'{model_name}_sample{sample_tag}.pkl')

    with open(out_model, 'wb') as f: pickle.dump(trained_model, f)
    with open(out_eval,  'wb') as f: pickle.dump(model_results, f)

    print(f"  Saved model  -> {out_model}")
    print(f"  Saved eval   -> {out_eval}")

    shutil.rmtree(partial_dir)
    print(f"  Deleted partial dir: {partial_dir}")

print("\nDone.")