import minedojo

# Uncomment to see more logs of the MineRL launch
#import coloredlogs
#import logging
#coloredlogs.install(logging.DEBUG)

env = minedojo.make(task_id='open-ended', image_size=(160,256))
env.reset()

done = False
while not done:
    ac = env.action_space.no_op()
    obs, reward, done, info = env.step(ac)
    env.render()
env.close()