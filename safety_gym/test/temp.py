import gym
import safety_gym
import numpy as np
import matplotlib.pyplot as plt

def stack_image(image):
    image = np.sum(image, axis=-1)
    # image = image[:,:, 2]
    new_image = np.zeros((image.shape[0], image.shape[1]))
    loc0, loc1 = np.where(image > 0)
    new_image[loc0, loc1] = 1
    return new_image


if __name__ == "__main__":
    env = gym.make("Safexp-MassGoal2-v0")
    env.observe_com = True
    env.robot_rot = 0
    obs = env.reset()
    print(env.world.robot_com())
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        action = np.array([0, 1])
        # import pdb
        # pdb.set_trace()
        next_obs, reward, done, info = env.step(action)
        # new_img = stack_image(next_obs)
        # plt.imshow(new_img)
        # plt.savefig('temp.jpg')
        # plt.cla()
        # plt.clf()
        env.render()
        if steps >= 1000:
            done = True
    print("done")