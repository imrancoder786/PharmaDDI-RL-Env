import requests
import json

s = requests.Session()
r1 = s.post('http://localhost:8000/reset', json={'task_name': 'easy_pair_check'})
d1 = r1.json()
print('Reset patient:', d1.get('observation', {}).get('patient_id'))
print('Task loaded:', d1.get('observation', {}).get('task_name'))

r2 = s.post('http://localhost:8000/step', json={'action': {'interactions_found': []}})
d2 = r2.json()
print('Step feedback:', d2.get('observation', {}).get('feedback'))
print('Step reward:', d2.get('reward'))
