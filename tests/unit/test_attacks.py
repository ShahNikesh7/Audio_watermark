#!/usr/bin/env python3
"""
Test script to verify all attack modules work correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from attacks import ATTACK_TYPES, simulate_attack

def test_attacks():
    """Test all available attack types"""
    print("Testing SoundSafe.ai Attack Suite")
    print("=" * 50)
    
    # Generate test audio
    audio = np.random.randn(44100)  # 1 second of random audio
    
    successful_attacks = []
    failed_attacks = []
    
    for attack_type in ATTACK_TYPES:
        try:
            result = simulate_attack(audio, attack_type, severity=0.5)
            if len(result) > 0:
                successful_attacks.append(attack_type)
                print(f"✓ {attack_type}")
            else:
                failed_attacks.append(attack_type)
                print(f"✗ {attack_type} - empty result")
        except Exception as e:
            failed_attacks.append(attack_type)
            print(f"✗ {attack_type} - error: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Results: {len(successful_attacks)}/{len(ATTACK_TYPES)} attacks successful")
    
    if failed_attacks:
        print(f"\nFailed attacks: {', '.join(failed_attacks)}")
    
    return len(failed_attacks) == 0

if __name__ == "__main__":
    success = test_attacks()
    exit(0 if success else 1)
