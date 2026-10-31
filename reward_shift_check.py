import random
from collections import defaultdict
from src.continual_reward_env import ContinualRewardEnv
from src.interactive_negotiation_env import Scenario

random.seed(17)
q = defaultdict(float)

def episode(objective, learn=False):
    stocks = tuple(random.randint(1, 3) for _ in range(3))
    values = tuple(tuple(random.randint(0, 5) for _ in range(3)) for _ in range(2))
    env = ContinualRewardEnv(Scenario(stocks, values), objective=objective, max_turns=6)
    obs = env.reset()
    path = []
    done = False
    while not done:
        agent = env.current_player
        o = obs[agent]
        offer = o["current_offer_for_self"]
        state = (agent, o["private_values"], stocks, tuple(offer) if offer else (-1,-1,-1), o["offer_from_other_agent"], o["turns_remaining"])
        acts = [{"type":"offer","allocation":stocks}, {"type":"offer","allocation":(0,0,0)}, {"type":"offer","allocation":tuple(s//2 for s in stocks)}]
        acts += [{"type":"accept"},{"type":"reject"}] if o["offer_from_other_agent"] else [{"type":"disagree"}]
        action = random.choice(acts) if learn and random.random() < .15 else max(acts, key=lambda a:q[(state,repr(a))])
        path.append((state,repr(action),agent))
        a,b,rewards,done = env.step(agent,action)
        obs = (a,b)
    if learn:
        for state,action,agent in path:
            key=(state,action)
            q[key] += .1*(rewards[agent]-q[key])
    return sum(rewards)/2, int(env.outcome=="deal")

def train(task,n):
    for _ in range(n): episode(task,True)

def evaluate(task,n=200):
    scores=[episode(task) for _ in range(n)]
    return round(sum(x[0] for x in scores)/n,3), round(sum(x[1] for x in scores)/n,3)

print("Training Task A...",flush=True)
train("self_interest",2000)
before=evaluate("self_interest")
print("Task A after A (reward, deal rate):",before,flush=True)
print("Training Task B...",flush=True)
train("fairness",2000)
after=evaluate("self_interest")
print("Task A after B (reward, deal rate):",after,flush=True)
print("Task B after B (reward, deal rate):",evaluate("fairness"),flush=True)
print("Task A change (reward, deal rate):",(round(after[0]-before[0],3),round(after[1]-before[1],3)),flush=True)
