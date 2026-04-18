import random
import time
import os
import sys

# ANSI color codes
RED     = "\033[91m"
YELLOW  = "\033[93m"
GREEN   = "\033[92m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

# Symbols: (display, color, weight, payout_multiplier)
SYMBOLS = [
    ("7️  ", RED,     2,  20),
    ("💎 ", CYAN,    3,  10),
    ("🍒 ", RED,     6,   5),
    ("🍋 ", YELLOW,  8,   3),
    ("🍊 ", YELLOW,  9,   2),
    ("🍇 ", MAGENTA, 10,  1.5),
    ("⭐ ", YELLOW,  12,  1),
]

SYMBOL_NAMES  = [s[0] for s in SYMBOLS]
SYMBOL_COLORS = [s[1] for s in SYMBOLS]
SYMBOL_WEIGHTS = [s[2] for s in SYMBOLS]
SYMBOL_PAYOUTS = [s[3] for s in SYMBOLS]

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def spin_reel():
    return random.choices(range(len(SYMBOLS)), weights=SYMBOL_WEIGHTS, k=1)[0]

def colored(text, color):
    return f"{color}{text}{RESET}"

def print_banner():
    print(colored("╔══════════════════════════════════════╗", YELLOW))
    print(colored("║", YELLOW) + colored("   🎰  PYTHON SLOT MACHINE  🎰      ", BOLD + WHITE) + colored("║", YELLOW))
    print(colored("╚══════════════════════════════════════╝", YELLOW))
    print()

def print_reels(r1, r2, r3, spinning=False):
    def cell(idx):
        sym   = SYMBOL_NAMES[idx]
        color = SYMBOL_COLORS[idx]
        return colored(f" {sym} ", color)

    if spinning:
        row = colored("???", DIM + WHITE)
        cells = f"  {row}  |  {row}  |  {row}  "
    else:
        cells = f" {cell(r1)} | {cell(r2)} | {cell(r3)}"

    print(colored("┌─────────────────────────────┐", WHITE))
    print(colored("│", WHITE) + "                             " + colored("│", WHITE))
    print(colored("│", WHITE) + cells + colored("│", WHITE))
    print(colored("│", WHITE) + "                             " + colored("│", WHITE))
    print(colored("└─────────────────────────────┘", WHITE))

def animate_spin(final_r1, final_r2, final_r3, steps=8):
    for i in range(steps):
        clear()
        print_banner()
        r1 = spin_reel() if i < steps - 2 else final_r1
        r2 = spin_reel() if i < steps - 1 else final_r2
        r3 = spin_reel()
        print_reels(r1, r2, r3)
        time.sleep(0.12)
    clear()
    print_banner()
    print_reels(final_r1, final_r2, final_r3)

def evaluate(r1, r2, r3, bet):
    if r1 == r2 == r3:
        multiplier = SYMBOL_PAYOUTS[r1]
        winnings   = bet * multiplier
        name       = SYMBOL_NAMES[r1]
        return winnings, f"JACKPOT! Three {name}! x{multiplier}"
    elif r1 == r2 or r2 == r3 or r1 == r3:
        matched = r1 if r1 == r2 or r1 == r3 else r2
        multiplier = SYMBOL_PAYOUTS[matched] * 0.5
        winnings   = bet * multiplier
        name       = SYMBOL_NAMES[matched]
        return winnings, f"Two {name}! x{multiplier:.1f}"
    else:
        return 0, "No match. Try again!"

def print_paytable():
    print(colored("\n  ╔══════ PAY TABLE ══════╗", DIM + WHITE))
    for sym, color, _, mult in SYMBOLS:
        bar = "█" * int(mult * 2)
        print(f"  {colored(sym, color)}  3x → {colored(f'x{mult}', GREEN)}  {colored(bar, DIM + GREEN)}")
    print(colored("  ╚═══════════════════════╝\n", DIM + WHITE))

def get_bet(balance):
    while True:
        try:
            raw = input(colored(f"  Enter bet (1–{int(balance)}) or 'q' to quit: ", CYAN)).strip().lower()
            if raw == "q":
                return None
            bet = float(raw)
            if bet < 1:
                print(colored("  Minimum bet is ₱1.", RED))
            elif bet > balance:
                print(colored("  Not enough balance!", RED))
            else:
                return bet
        except ValueError:
            print(colored("  Enter a valid number.", RED))

def main():
    balance = 100.0
    spins   = 0
    wins    = 0

    clear()
    print_banner()
    print(colored("  Welcome! You start with ₱100.\n", WHITE))
    print_paytable()
    input(colored("  Press Enter to start spinning...", DIM))

    while balance > 0:
        clear()
        print_banner()

        # Stats bar
        stats = f"  Balance: {colored(f'₱{balance:.2f}', GREEN)}   Spins: {colored(str(spins), CYAN)}   Wins: {colored(str(wins), YELLOW)}"
        print(stats)
        print()

        print_reels(0, 0, 0, spinning=False)
        print()

        bet = get_bet(balance)
        if bet is None:
            break

        r1, r2, r3 = spin_reel(), spin_reel(), spin_reel()
        animate_spin(r1, r2, r3)

        winnings, message = evaluate(r1, r2, r3, bet)
        balance -= bet
        balance += winnings
        spins   += 1

        if winnings > 0:
            wins += 1
            print(colored(f"\n  🎉  {message}", BOLD + GREEN))
            print(colored(f"  You won ₱{winnings:.2f}!", GREEN))
        else:
            print(colored(f"\n  😔  {message}", YELLOW))
            print(colored(f"  You lost ₱{bet:.2f}.", RED))

        print(colored(f"  Balance: ₱{balance:.2f}", WHITE))
        print()

        if balance <= 0:
            print(colored("  💸 You're out of money! Game over.", BOLD + RED))
            break

        again = input(colored("  Spin again? (Enter / q): ", DIM)).strip().lower()
        if again == "q":
            break

    # Final summary
    clear()
    print_banner()
    print(colored("  ═══════ GAME OVER ═══════\n", YELLOW))
    print(f"  Total spins : {colored(str(spins), CYAN)}")
    print(f"  Total wins  : {colored(str(wins), GREEN)}")
    print(f"  Final balance: {colored(f'₱{balance:.2f}', YELLOW)}")
    net = balance - 100
    color = GREEN if net >= 0 else RED
    sign  = "+" if net >= 0 else ""
    print(f"  Net result  : {colored(f'{sign}₱{net:.2f}', color)}")
    print(colored("\n  Thanks for playing! 🎰\n", BOLD + WHITE))

if __name__ == "__main__":
    main()
