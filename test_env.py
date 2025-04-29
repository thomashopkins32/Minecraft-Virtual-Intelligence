from craftground import make

env = make(verbose_jvm=True)

env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    _ = env.step(action)
    env.render()

env.close()
