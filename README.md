# smolbrain

Here's how to run it in Termux step by step:

**1. Install Termux**
Get it from F-Droid (recommended) or the Play Store.

**2. Update packages and install Python**
```bash
pkg update && pkg upgrade
pkg install python
```

**3. Install NumPy**
```bash
pip install numpy
```

**4. Transfer the file to your phone**
A few options:
- **Easiest:** Upload `nn_slot_machine.py` to Google Drive, then in Termux:
  ```bash
  pkg install termux-api
  ```
  Or just download it directly with `curl` if you have a link.

- **Via USB:** Enable USB file transfer on your phone, copy the file to your Downloads folder, then in Termux:
  ```bash
  cp /sdcard/Download/nn_slot_machine.py .
  ```

- **Quick copy-paste:** If the file is small enough, just `nano nn_slot_machine.py`, paste the contents, then `Ctrl+X → Y → Enter` to save.

**5. Run it**
```bash
python nn_slot_machine.py
```

---

**Heads up for Termux specifically:**
- The emoji symbols (🎰, 💎, 🍒, etc.) should render fine on modern Android
- ANSI colors work in Termux by default
- NumPy may take a minute to install — that's normal
- If you see a `TERM environment variable not set` warning, just ignore it — it's cosmetic and the script runs fine