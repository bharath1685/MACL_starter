import json, random
from pathlib import Path
from interactive_negotiation_env import NegotiationEnv, Scenario

rng=random.Random(17)
out=Path("outputs/interactive_trajectories.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists(): raise FileExistsError(out)
counts={}
with out.open("w") as f:
 for ep in range(100):
  stocks=tuple(rng.randint(1,4) for _ in range(3))
  values=tuple(tuple(rng.randint(1,5) for _ in range(3)) for _ in range(2))
  env=NegotiationEnv(Scenario(stocks,values),max_turns=10,gap_weight=0.0)
  env.reset(seed=17+ep)
  traj=[]
  while not env.done:
   a=env.current_player; obs=env.observe(a)
   if obs["offer_from_other_agent"]: action={"type":"accept"}
   elif obs["offer_exists"]: action={"type":"reject"}
   else: action={"type":"offer","allocation":[(s+1)//2 for s in stocks]}
   traj.append({"agent":a,"observation":obs,"action":action})
   env.step(a,action)
  f.write(json.dumps({"episode":ep,"stocks":stocks,"private_values":values,"trajectory":traj,"outcome":env.outcome,"rewards":env.rewards,"turns":env.turns})+"\n")
  counts[env.outcome]=counts.get(env.outcome,0)+1
print("Saved 100 scripted pilot episodes:",out)
print("Outcomes:",counts)
