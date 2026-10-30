import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from src.edge.router import route_intent

tests = [
    'hello',
    'what time is it',
    'what is today date',
    'calculate 48 times 12',
    'what is 250 plus 375',
    'what is 144 divided by 12',
    'check battery',
    'ram usage',
    'cpu usage',
    'who are you',
    'tell me a story about dragons',
    '',
]

print('=== ROUTER STANDALONE TEST ===')
for t in tests:
    result = route_intent(t)
    status = 'FAST ROUTER' if result else '-> LLM'
    display = str(result)[:80] if result else 'None (forwarded to LLM)'
    inp = repr(t[:45])
    print(f'[{status}]')
    print(f'  Input: {inp}')
    print(f'  Reply: {display}')
    print()
