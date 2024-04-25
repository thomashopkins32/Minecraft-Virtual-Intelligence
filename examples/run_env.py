import minedojo  # type: ignore

env = minedojo.make(task_id="open-ended", image_size=(160, 256))
obs = env.reset()
done = False
while not done:
    full_action = env.action_space.no_op()
    obs, reward, done, info = env.step(full_action)
    env.render()
env.close()
