import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from simulator.backboard import Backboard
from simulator.ball import Ball
from time import sleep

class Simulator:
    def __init__(self) -> None:
        self.backboard = Backboard(orig=(0, 15, 15))
        self.ball = Ball(orig=(0, 0, 0), velocity=(0.2, 5, 20))
        self.result = Ball()
        self.target = (0, 10, 10)

        self.fig = plt.figure()
        self.fig.suptitle('Deflect the Ball Simulation')
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_zlim(-20, 20)

        self.fig.subplots_adjust(bottom=0.25)

        self.ax_posx = self.fig.add_axes([0.05, 0.1, 0.35, 0.05])
        self.ax_posx.grid(False)
        self.ax_posx.set_title('Position')
        self.x_slider = Slider(
            ax=self.ax_posx,
            label='X',
            valmin=-20,
            valmax=20,
            valinit=0
        )

        self.ax_posy = self.fig.add_axes([0.05, 0.05, 0.35, 0.05])
        self.ax_posy.grid(False)
        self.y_slider = Slider(
            ax=self.ax_posy,
            label='Y',
            valmin=-20,
            valmax=12,
            valinit=0
        )

        self.ax_posz = self.fig.add_axes([0.05, 0.0, 0.35, 0.05])
        self.ax_posz.grid(False)
        self.z_slider = Slider(
            ax=self.ax_posz,
            label='Z',
            valmin=0,
            valmax=20,
            valinit=0
        )

        self.x_slider.on_changed(self.update)
        self.y_slider.on_changed(self.update)
        self.z_slider.on_changed(self.update)

        self.ax_velx = self.fig.add_axes([0.55, 0.1, 0.35, 0.05])
        self.ax_velx.grid(False)
        self.ax_velx.set_title('Velocity')
        self.vx_slider = Slider(
            ax=self.ax_velx,
            label='X',
            valmin=-20,
            valmax=20,
            valinit=0.2
        )
        self.ax_vely = self.fig.add_axes([0.55, 0.05, 0.35, 0.05])
        self.ax_vely.grid(False)
        self.vy_slider = Slider(
            ax=self.ax_vely,
            label='Y',
            valmin=0.01,
            valmax=20,
            valinit=5
        )
        self.ax_velz = self.fig.add_axes([0.55, 0.00, 0.35, 0.05])
        self.ax_velz.grid(False)
        self.vz_slider = Slider(
            ax=self.ax_velz,
            label='Z',
            valmin=0.01,
            valmax=20,
            valinit=20
        )

        self.vx_slider.on_changed(self.update)
        self.vy_slider.on_changed(self.update)
        self.vz_slider.on_changed(self.update)

    def show(self) -> None:
        self.draw()
        plt.show()

    def draw(self) -> None:
        #
        # Calculate how the backboard needs to rotate to get the ball to the target
        #
        t_target = self.calc_backboard_reflection()
            
        #
        # Plot the backboard
        #
        brd_x, brd_y, brd_z = self.backboard.draw()
        self.ax.plot_surface(brd_x, brd_y, brd_z, alpha=0.5, color='blue')
        self.ax.plot([self.backboard.orig[0]], [self.backboard.orig[1]], [self.backboard.orig[2]], 'ro')

        #
        # Plot the ball trajectory
        #

        # Plot a point for the ball origin
        self.ax.plot(self.ball.orig[0], self.ball.orig[1], self.ball.orig[2], 'ro')

        # Get the time until the ball hits the backboard
        t_total = (self.backboard.orig[1] - self.ball.orig[1]) / self.ball.v[1]

        # Get the number of samples to make the graph look continuous
        samples = abs(int(t_total * 20))

        # Get the position at each time step
        bal_xdata = []
        bal_ydata = []
        bal_zdata = []
        for t in np.linspace(0, t_total, samples):
            self.ball.update(t)
            bal_x, bal_y, bal_z = self.ball.position()
            bal_xdata.append(bal_x)
            bal_ydata.append(bal_y)
            bal_zdata.append(bal_z)
            # If ball hits the ground before reaching the backboard,
            # reflect that in the data
            if bal_z <= 0 and t > 0:
                break
        self.ax.plot(bal_xdata, bal_ydata, bal_zdata, 'g-')

        #
        # Draw the target
        # 
        self.ax.plot(self.target[0], self.target[1], self.target[2], 'go')

        # Print the target trajectory
        if t_target is not None:
            samples = abs(int(t_target) * 20)
            r_xdata = []
            r_ydata = []
            r_zdata = []
            for t in np.linspace(0, t_target, num=samples):
                self.result.update(t)
                r_x, r_y, r_z = self.result.position()
                r_xdata.append(r_x)
                r_ydata.append(r_y)
                r_zdata.append(r_z)
            self.ax.plot(r_xdata, r_ydata, r_zdata, 'b-')
    
    def calc_backboard_reflection(self) -> float:
        # First, get the time until the ball will reach the backboard and target
        tg_coord = (self.target[1] + 0.5)
        t_target = (tg_coord - self.ball.orig[1]) / self.ball.v[1]
        t_backbrd = (self.backboard.orig[1] - self.ball.orig[1]) / self.ball.v[1]

        # Now, for each timestep between those two values, calculate the
        # how the backboard normal needs to change to reflect the ball to
        # the target and record the amount of movement to reach that normal
        solutions = []
        idx = 0
        for t in np.linspace(t_target, t_backbrd, num=100):
            # Get the position of the ball at that timestep
            self.ball.update(t)
            x_init, y_init, z_init = self.ball.position()

            # Return none if ball is outside bounds
            if ((z_init > self.backboard.orig[2] + self.backboard.h/2) or
                (z_init < self.backboard.orig[2] - self.backboard.h/2) or
                (x_init > self.backboard.orig[0] + self.backboard.w/2) or
                (x_init < self.backboard.orig[0] - self.backboard.w/2)):
                return None

            # 
            # Solve for the needed velocity components to get to the target
            # 

            # Start by applying quadratic formula to solve for vz and t
            # NOTE: b^2 - 4ac >= 0 for real solutions, so:
            #       b = sqrt(4ac)
            # NOTE: The equation zf = zi + vz*t - 0.5g*t^2 needs to be 
            #       rearranged to:
            #       0 = -0.5g*t^2 + vz*t + (zi - zf) 
            #       to use the quadratic formula
            c = (z_init - self.target[2])   # c = zi - zf
            a = -0.5*self.ball.g            # a = -0.5*g
            b = np.sqrt(-4*a*c)             # b = vz
            if not np.isnan(b):
                roots = np.roots([a, b, c])
                t_tg = roots[roots > 0][0]
                vz = b

                # Get vx and vy
                vx = (self.target[0] - x_init) / t_tg
                vy = (self.target[1] - y_init) / t_tg

                # Solve for the new normal vector that the plane should be
                # Based on: vr = vi - 2*(vi.n)n where n is a normal vector
                vx_b, vy_b, vz_b = self.ball.velocity()
                vi = np.array([vx_b, vy_b, vz_b])
                vr = np.array([vx, vy, vz])
                new_norm = vi - vr
                new_norm = new_norm / np.linalg.norm(new_norm)

                solutions.append({
                    'impact_t': t, 
                    'target_t': t_tg, 
                    'initial_velocity': [vx, vy, vz],
                    'initial_position': [x_init, y_init, z_init],
                    'normal': new_norm
                })
            
        # Now rank the solutions and find the optimal one
        # based on if the backboard can move to that position
        # and for the least amount of movement
        rankings = []
        for s in solutions:
            normal = s['normal']
            pi = np.array(s['initial_position'])

            # Make sure the plane can reach the ball
            d = -self.backboard.orig.dot(normal)
            if (pi.dot(normal) + d) == 0:
                # If so, rank based on movement
                a = self.backboard.norm[0] - normal[0]
                b = self.backboard.norm[1] - normal[1]
                c = self.backboard.norm[2] - normal[2]
                ranking = abs(a + b + c)
                rankings.append(ranking)
            else:
                # Otherwise, set the rank to be unfavored
                rankings.append(100.0)
        
        # Now get the minimal ranking and use that for our new normal
        min_rank = min(rankings)
        idx = rankings.index(min_rank)
        
        self.result.v[0] = solutions[idx]['initial_velocity'][0]
        self.result.v[1] = solutions[idx]['initial_velocity'][1]
        self.result.v[2] = solutions[idx]['initial_velocity'][2]

        self.result.orig[0] = solutions[idx]['initial_position'][0]
        self.result.orig[1] = solutions[idx]['initial_position'][1]
        self.result.orig[2] = solutions[idx]['initial_position'][2]

        self.backboard.norm[0] = solutions[idx]['normal'][0]
        self.backboard.norm[1] = solutions[idx]['normal'][1]
        self.backboard.norm[2] = solutions[idx]['normal'][2]

        return solutions[idx]['target_t']

    def old_calc_backboard_reflection(self) -> float:
        # First, get the time until the ball will reach the backboard
        t_impact = (self.target[1] - self.ball.orig[1]) / self.ball.v[1]

        # Next, get all the initial coords from that time
        self.ball.update(t_impact)
        x_init, y_init, z_init = self.ball.position()

        # Make sure the z and x coords are within backboard bounds
        if ((z_init > self.backboard.orig[2] + self.backboard.h/2) or
            (z_init < self.backboard.orig[2] - self.backboard.h/2) or
            (x_init > self.backboard.orig[0] + self.backboard.w/2) or
            (x_init < self.backboard.orig[0] - self.backboard.w/2)):
            print('Out of Bounds')
            return None
        
        #
        # Calculate the velocity vector needed to reach the target
        # 

        # Assume that we will just be negating the y velocity component
        vy_reflect = -self.ball.v[1]

        # Get the time to reach the target y from there
        t_tg = (self.target[1] - y_init) / vy_reflect

        # Get the x and z velocity components from there
        vx_reflect = (self.target[0] - x_init) / t_tg
        vz_reflect = (self.target[2] - z_init) / t_tg

        # Calculate the normal needed for the velocity vector
        vi = np.array([vx_reflect, vy_reflect, vz_reflect])
        new_norm = self.ball.v - vi
        if np.linalg.norm(new_norm) == 0:
            return None
        new_norm = new_norm / np.linalg.norm(new_norm)

        # Set the new normal for the backboard
        self.backboard.norm[0] = new_norm[0]
        self.backboard.norm[1] = new_norm[1]
        self.backboard.norm[2] = new_norm[2]

        # Set the resulting ball reflection values
        self.result.orig[0] = x_init
        self.result.orig[1] = y_init
        self.result.orig[2] = z_init
        self.result.v[0] = vx_reflect
        self.result.v[1] = vy_reflect
        self.result.v[2] = vz_reflect

        # Return the time to the target
        return t_tg


    def update(self, _) -> None:
        self.ball.orig[0] = self.x_slider.val
        self.ball.orig[1] = self.y_slider.val
        self.ball.orig[2] = self.z_slider.val
        self.ball.v[0] = self.vx_slider.val
        self.ball.v[1] = self.vy_slider.val
        self.ball.v[2] = self.vz_slider.val
        self.ax.clear()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_zlim(-20, 20)
        self.draw()
        self.fig.canvas.draw_idle()