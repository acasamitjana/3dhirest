import numpy as np
import torch


def predict(generator_data, model, prediction_size, device, da_model=None):
    num_elements = len(generator_data.dataset)
    num_batches = len(generator_data)
    batch_size = generator_data.batch_size

    ref_image = np.zeros((num_elements,) + prediction_size)
    flo_image = np.zeros((num_elements,) + prediction_size)
    reg_image = np.zeros((num_elements,) + prediction_size)
    flow_image = np.zeros((num_elements, 2) + prediction_size)
    rid_list = []

    with torch.no_grad():
        for batch_idx, (ref, flo, ref_mask, flo_mask, _, _, angle, nonlinear_field, rid) in enumerate(generator_data):

            ref = ref.to(device)
            flo = flo.to(device)
            nonlinear_field = [non.to(device) for non in nonlinear_field]
            if da_model is not None:
                fliplr = 0  # np.random.rand(1)
                flipud = 0  # np.random.rand(1)
                # ref = da_model.transform(ref, angle[0], nonlinear_field[0], flipud, fliplr)
                flo = da_model.transform(flo, angle[1], nonlinear_field[1], flipud, fliplr)

            r, f, v = model(flo, ref)

            start = batch_idx * batch_size
            end = start + batch_size
            if batch_idx == num_batches - 1:
                end = num_elements
            ref_image[start:end] = ref[:, 0].cpu().detach().numpy()
            flo_image[start:end] = flo[:, 0].cpu().detach().numpy()
            reg_image[start:end] = r[:, 0].cpu().detach().numpy()
            flow_image[start:end] = np.squeeze(f.cpu().detach().numpy())
            rid_list.extend(rid)

    return ref_image, flo_image, reg_image, flow_image, rid_list
