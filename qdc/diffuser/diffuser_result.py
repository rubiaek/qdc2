from qdc.diffuser.field import Field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import cv2
from matplotlib.widgets import Button


class DiffuserResult:
    def __init__(self):
        self._SPDC_fields_E = None
        self._SPDC_fields_wl = None
        self.SPDC_fields = None
        self.SPDC_delta_lambdas = None
        self._classical_fields_E = None
        self._classical_fields_wl = None
        self.classical_fields = None
        self.classical_delta_lambdas = None
        self.wavelengths = None
        self.SPDC_PCCs = None
        self.SPDC_incoherent_sum = None
        self.classical_PCCs = None
        self.classical_incoherent_sum = None

        self.classical_ff_method = ''
        self.classical_xs = []
        self.classical_ys = []


    def _populate_res_SPDC(self, roi=None):
        if roi is None:
            roi = np.index_exp[:]
        self.SPDC_fields = []
        for field_E, wl in zip(self._SPDC_fields_E, self._SPDC_fields_wl):
            self.SPDC_fields.append(Field(self.x, self.y, wl, field_E))

        self.SPDC_fields = np.array(self.SPDC_fields)

        PCCs = []
        f0 = self.SPDC_fields[0]
        for f in self.SPDC_fields:
            PCC = np.corrcoef(f0.I[roi].ravel(), f.I[roi].ravel())[0, 1]
            PCCs.append(PCC)
        self.SPDC_PCCs = np.array(PCCs)

        final_I = np.zeros_like(self.SPDC_fields[0].I)
        for field in self.SPDC_fields:
            final_I += field.I
        self.SPDC_incoherent_sum = final_I

    def fix_grids(self, fields):
        from scipy.interpolate import RegularGridInterpolator
        new_fs = []
        max_Xs = [f.x.max() for f in fields]
        # low wavelength means smaller total size of grid, and I want to only interpolate, not extrapolate,
        # so down-sizing the larger grids to the smallest one
        assert min(max_Xs) == max_Xs[0]
        global_x = fields[0].x
        global_y = fields[0].y
        global_XX, global_YY = np.meshgrid(global_x, global_y)
        points = np.array([global_XX.flatten(), global_YY.flatten()]).T

        for f in fields:
            interp_func = RegularGridInterpolator((f.x, f.y), f.E)
            interpolated = interp_func(points).reshape(global_XX.shape)
            new_fs.append(Field(global_x, global_y, f.wl, interpolated))

        return new_fs

    def _populate_res_classical(self, roi=None, fix_grids=True):
        if roi is None:
            roi = np.index_exp[:]
        self.classical_fields = []

        if self.classical_ff_method == 'fft':
            for field_E, wl, x, y in zip(self._classical_fields_E, self._classical_fields_wl,
                                         self.classical_xs, self.classical_ys):
                self.classical_fields.append(Field(x, y, wl, field_E))
            if fix_grids:
                self.classical_fields = self.fix_grids(self.classical_fields)
        else:
            for field_E, wl in zip(self._classical_fields_E, self._classical_fields_wl):
                self.classical_fields.append(Field(self.x, self.y, wl, field_E))

        self.classical_fields = np.array(self.classical_fields)

        PCCs = []
        f0 = self.classical_fields[0]
        for f in self.classical_fields:
            PCC = np.corrcoef(f0.I[roi].ravel(), f.I[roi].ravel())[0, 1]
            PCCs.append(PCC)
        self.classical_PCCs = np.array(PCCs)

        final_I = np.zeros_like(self.classical_fields[0].I)
        for field in self.classical_fields:
            final_I += field.I
        self.classical_incoherent_sum = final_I

    # contrast =  np.std(data) / np.mean(data)

    def show_interactive(self, SPDC=True, save_mp4_to=None, fps=3):
        # Select frames
        if SPDC:
            frames = [f.I for f in self.SPDC_fields]
            wls = self._SPDC_fields_wl
        else:
            frames = [f.I for f in self.classical_fields]
            wls = self._classical_fields_wl

        # Save MP4 with text overlay if requested
        if save_mp4_to is not None:
            viridis = plt.get_cmap('viridis')
            video_frames = [np.uint8(viridis(frame / frame.max())[:, :, :3] * 255) for frame in frames]
            height, width = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_mp4_to, fourcc, fps, (width, height))

            for i, frame in enumerate(video_frames):
                text = f"wl {wls[i]*1e9:.2f} nm"  # Customize your text here
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 5
                thickness = 5
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = 120
                frame_with_text = cv2.putText(
                    frame.copy(), text, (text_x, text_y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA
                )
                out.write(cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR))

            out.release()
            print(f"Video saved as {save_mp4_to}")

        # Set up the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        img_display = ax.imshow(frames[0], cmap='viridis')
        ax.set_title( f"wl {wls[0]*1e9:.2f} nm")
        plt.ion()  # Interactive mode on

        # Button positions
        ax_play = plt.axes([0.1, 0.01, 0.15, 0.05])
        ax_prev = plt.axes([0.3, 0.01, 0.15, 0.05])
        ax_next = plt.axes([0.5, 0.01, 0.15, 0.05])

        # Create buttons
        play_button = Button(ax_play, 'Play/Pause')
        prev_button = Button(ax_prev, 'Previous')
        next_button = Button(ax_next, 'Next')

        # State variables
        class State:
            def __init__(self):
                self.playing = False
                self.frame_idx = 0

        state = State()

        def update_frame(idx):
            state.frame_idx = idx % len(frames)
            img_display.set_data(frames[state.frame_idx])
            ax.set_title( f"wl {wls[idx]*1e9:.2f} nm")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()  # Ensure events are processed

        def on_play_clicked(event):
            state.playing = not state.playing
            play_button.label.set_text("Pause" if state.playing else "Play")
            while state.playing and state.frame_idx < len(frames) - 1:
                state.frame_idx += 1
                update_frame(state.frame_idx)
                plt.pause(0.033)  # ~30 FPS
            if state.frame_idx >= len(frames) - 1:
                state.playing = False
                play_button.label.set_text("Play")
                state.frame_idx = 0
                update_frame(state.frame_idx)

        def on_next_clicked(event):
            state.playing = False
            play_button.label.set_text("Play")
            if state.frame_idx < len(frames) - 1:
                update_frame(state.frame_idx + 1)
            fig.canvas.draw_idle()  # Force redraw

        def on_prev_clicked(event):
            state.playing = False
            play_button.label.set_text("Play")
            if state.frame_idx > 0:
                update_frame(state.frame_idx - 1)
            fig.canvas.draw_idle()  # Force redraw

        # Connect callbacks explicitly
        play_button.on_clicked(on_play_clicked)
        prev_button.on_clicked(on_prev_clicked)
        next_button.on_clicked(on_next_clicked)

        # Store button references to prevent garbage collection
        self._buttons = [play_button, prev_button, next_button]  # Keep alive

        # Display figure
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.1)  # Brief pause to ensure rendering

    def plot_PCCs_SPDC(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.SPDC_delta_lambdas*1e9, self.SPDC_PCCs, '*--', label='PCC')
        ax.set_xlabel('$\Delta\lambda$ [nm]')
        ax.set_ylabel('PCC')
        ax.set_title('SPDC experiment')
        # ax.figure.show()

    def show_incoherent_sum_SPDC(self, ax=None):
        Field(self.x, self.y, self.wavelengths[0], np.sqrt(self.SPDC_incoherent_sum)).show(title='Incoherent sum of all wavelengths SPDC', ax=ax)

    def plot_PCCs_classical(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.classical_delta_lambdas*1e9, self.classical_PCCs, '*--', label='PCC')
        ax.set_xlabel('$\Delta\lambda$ [nm]')
        ax.set_ylabel('PCC')
        ax.set_title('Classical experiment')
        # ax.figure.show()

    def show_incoherent_sum_classical(self, ax=None):
        Field(self.x, self.y, self.wavelengths[0], np.sqrt(self.classical_incoherent_sum)).show(title='Incoherent sum of all wavelengths classical', ax=ax)

    def show(self, sq_D):
        fig, axes = plt.subplots(2, 2, figsize=(11,10))
        self.show_incoherent_sum_SPDC(axes[0, 0])
        self.show_incoherent_sum_classical(axes[0, 1])
        x_c = y_c = self.Nx // 2
        x_c = y_c = 0
        rect = patches.Rectangle(
            (x_c - sq_D, y_c - sq_D),  # Bottom-left corner
            sq_D*2, sq_D*2,  # Width, Height
            edgecolor='white',  # Border color
            facecolor='none',  # Fill color
            linewidth=0.5  # Border width
        )
        rect2 = patches.Rectangle(
            (x_c - sq_D, y_c - sq_D),  # Bottom-left corner
            sq_D * 2, sq_D * 2,  # Width, Height
            edgecolor='white',  # Border color
            facecolor='none',  # Fill color
            linewidth=0.5  # Border width
        )
        axes[0, 0].add_patch(rect)
        axes[0, 1].add_patch(rect2)
        self.plot_PCCs_SPDC(axes[1, 0])
        self.plot_PCCs_classical(axes[1, 1])

    def show_diffuser(self):
        fig, ax = plt.subplots()
        pcm = ax.imshow(self.diffuser_mask, extent=[self.x[0] * 1e3, self.x[-1] * 1e3, self.y[0] * 1e3, self.y[-1] * 1e3],
                        cmap='viridis', origin='lower')
        fig.colorbar(pcm, ax=ax, label='Phase [rad]')
        ax.set_title("Single Diffuser Phase")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        # fig.show()

    def saveto(self, path, save_fields=False):
        d = copy.deepcopy(self.__dict__)
        d.pop('SPDC_fields')
        if not save_fields:
            d.pop('_SPDC_fields_E')
        np.savez(path, **d)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)
