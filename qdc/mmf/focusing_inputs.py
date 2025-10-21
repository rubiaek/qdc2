from qdc.mmf.fiber import Fiber

def get_required_input_SPDC(fiber, focus_size, X0, Y0):
    fiber.set_input_gaussian(sigma=focus_size, X0=X0, Y0=Y0)

    # Find required input with OPC-like approach 
    E_end0 = fiber.propagate(show=False, free_mode_matrix=False)
    fiber.profile_0 = E_end0
    E_end = fiber.propagate(show=False, free_mode_matrix=True)
    required_input = E_end.conj()
    return required_input

def get_required_input_classical(fiber, focus_size, X0, Y0):
    fiber.set_input_gaussian(sigma=focus_size, X0=X0, Y0=Y0)
    E_end = fiber.propagate(show=False, free_mode_matrix=True)
    required_input = E_end.conj()
    return required_input

def get_required_input_SPDC_before_fiber(fiber, focus_size, X0, Y0, laser_X0, laser_Y0, laser_focus_size=0.6):
    assert isinstance(fiber, Fiber), "fiber must be a Fiber object"
    # Assuming the input to the AWP simulation is the laser Gaussian at X0, Y0
    fiber.set_input_gaussian(sigma=focus_size, X0=X0, Y0=Y0)
    fiber.L *= -1 # Hack for doing "backward propagation" and then reverting 
    field_backward = fiber.propagate(show=False, free_mode_matrix=True).copy()
    fiber.L *= -1 

    fiber.set_input_gaussian(sigma=laser_focus_size, X0=laser_X0, Y0=laser_Y0)
    field_forward = fiber.propagate(show=False, free_mode_matrix=True).copy()

    overlap = field_backward.conj() * field_forward
    
    return overlap


    # forward + SLM = backward 
    # SLM = backward - forward 