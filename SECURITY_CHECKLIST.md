# Security Checklist for Open Source Release

This document outlines the security measures taken to prepare this codebase for open source release.

## ✅ Issues Fixed

### 1. API Keys Removed
- **Files affected**: `train_segment_diffusion.py`, `train_latent_model.py`
- **Issue**: Hardcoded WandB API keys
- **Fix**: Removed hardcoded API keys, now reads from environment variables
- **Action required**: Set `WANDB_API_KEY` environment variable before running

### 2. Personal File Paths Replaced
- **Files affected**: 
  - `dataset_utils/ae_dataset.py`
  - `train_diffusion.sh`
  - `saved_diff_models/full_roc_utt_jsonl/2025-07-19_18-23-29/args.json`
  - `diffusion/sentence_denoising_diffusion.py`
- **Issue**: Hardcoded paths containing username `/home/xz479/`
- **Fix**: Replaced with relative paths and placeholders

## ⚠️ Files to Remove/Clean Before Open Sourcing

### 1. WandB Log Files (Contains Personal Information)
The following directories contain system information including hostname "dev-gpu-xz479":
```
wandb/
├── debug-internal.log
├── debug.log
├── debug-cli.xz479.log
├── latest-run/
├── offline-run-*/
```

**Recommendation**: Delete the entire `wandb/` directory before open sourcing.

### 2. Saved Model Directories
```
saved_diff_models/
saved_latent_models/
```
These may contain model checkpoints and configuration files with personal paths.

**Recommendation**: Either delete or carefully review these directories.

## 🔧 Setup Instructions for Users

### Environment Variables Required
```bash
export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_PROJECT="your_project_name"
```

### Update Paths in Scripts
Users will need to update the following placeholders:
- `saved_latent_models/your_model_name_here` in `train_diffusion.sh`
- Any dataset paths to point to their local datasets

## 🚀 Clean Release Commands

Before creating your release, run these commands:

```bash
# Remove WandB logs
rm -rf wandb/

# Remove saved models (optional - review first)
rm -rf saved_diff_models/
rm -rf saved_latent_models/

# Create .gitignore for future commits
echo "wandb/" >> .gitignore
echo "saved_*_models/" >> .gitignore
echo "*.pt" >> .gitignore
echo "*.pth" >> .gitignore
```

## 📝 Additional Recommendations

1. **Review datasets**: Ensure datasets in `datasets/` don't contain sensitive information
2. **Check requirements.txt**: Verify all dependencies are appropriate for public release
3. **Add LICENSE file**: Choose an appropriate open source license
4. **Update README.md**: Add installation and usage instructions
5. **Environment setup**: Consider adding a `setup.py` or `pyproject.toml` file

## ✅ Verification Checklist

- [ ] All API keys removed from source code
- [ ] No personal file paths in any `.py`, `.sh`, or `.json` files
- [ ] WandB directory deleted
- [ ] Saved model directories reviewed/cleaned
- [ ] Environment variable setup documented
- [ ] Installation instructions provided
- [ ] License file added