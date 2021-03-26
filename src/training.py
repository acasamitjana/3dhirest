

def train(model,
          optimizer,
          device,
          generator_train,
          epoch,
          loss_function_dict,
          loss_weight_dict,
          callbacks,
          da_model,
          mask_flag=False
          ):

    total_iter = 0
    N = len(generator_train.dataset)
    model = model.to(device)

    for batch_idx, data_list in enumerate(generator_train):

        ref_image, flo_image = data_list[0].to(device), data_list[1].to(device)
        ref_mask, flo_mask = data_list[2].to(device), data_list[3].to(device)
        nonlinear_field = [nlf.to(device) for nlf in data_list[7]]
        angle = data_list[6]

        total_iter += len(ref_image)
        model.zero_grad()

        if da_model is not None:
            fliplr = 0#np.random.rand(1)
            flipud = 0#np.random.rand(1)
            ref_image = da_model.transform(ref_image, angle[0], nonlinear_field[0], flipud, fliplr)
            ref_mask = da_model.transform(ref_mask, angle[0], nonlinear_field[0], flipud, fliplr)
            flo_image = da_model.transform(flo_image, angle[1], nonlinear_field[1], flipud, fliplr)

        reg_image, flow_image, v_image = model(flo_image, ref_image)

        loss_dict = {}
        if mask_flag:
            loss_dict['registration'] = loss_function_dict['registration'](reg_image, ref_image, ref_mask)
        else:
            loss_dict['registration'] = loss_function_dict['registration'](reg_image, ref_image)

        loss_dict['registration_smoothness'] = loss_function_dict['registration_smoothness'](flow_image)
        for k,v in loss_dict.items():
            loss_dict[k] = loss_weight_dict[k] * v

        total_loss = sum([l for l in loss_dict.values()])

        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        # plot_grad_flow(model.named_parameters(), save_dir='model_reg')
        optimizer.step()

        log_dict = {**{'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                    **{'loss': total_loss.item()}
                    }

        for cb in callbacks:
            cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)


    return callbacks


def train_bidirectional(model,
                        optimizer,
                        device,
                        generator_train,
                        epoch,
                        loss_function_dict,
                        loss_weight_dict,
                        callbacks,
                        da_model,
                        mask_flag=True
                        ):

    total_iter = 0
    N = len(generator_train.dataset)
    rid_epoch_list = []
    for batch_idx, data_list in enumerate(generator_train):

        ref_image, flo_image = data_list[0].to(device), data_list[1].to(device)
        ref_mask, flo_mask = data_list[2].to(device), data_list[3].to(device)
        nonlinear_field = [nlf.to(device) for nlf in data_list[7]]
        angle = data_list[6]
        rid_epoch_list.extend(data_list[-1])

        total_iter += len(ref_image)
        model.zero_grad()

        if da_model is not None:
            fliplr = 0#np.random.rand(1)
            flipud = 0#np.random.rand(1)
            ref_image = da_model.transform(ref_image, angle[0], nonlinear_field[0], flipud, fliplr)
            flo_image = da_model.transform(flo_image, angle[1], nonlinear_field[1], flipud, fliplr)
            ref_mask = da_model.transform(ref_mask, angle[0], nonlinear_field[0], flipud, fliplr)
            flo_mask = da_model.transform(flo_mask, angle[1], nonlinear_field[1], flipud, fliplr)

        reg_flo_image, flow_image, v_image = model(flo_image, ref_image)
        reg_ref_image = model.predict(ref_image, -v_image, diffeomorphic=True)

        loss_dict = {}
        if mask_flag:
            loss_dict['registration'] = loss_function_dict['registration'](reg_ref_image, flo_image, flo_mask)
            loss_dict['registration'] += loss_function_dict['registration'](reg_flo_image, ref_image, ref_mask)
            loss_dict['registration'] = 0.5 * loss_dict['registration']

        else:
            loss_dict['registration'] = loss_function_dict['registration'](reg_ref_image, flo_image)
            loss_dict['registration'] += loss_function_dict['registration'](reg_flo_image, ref_image)
            loss_dict['registration'] = 0.5 * loss_dict['registration']

        loss_dict['registration_smoothness'] = loss_function_dict['registration_smoothness'](v_image)

        for k, v in loss_dict.items():
            loss_dict[k] = loss_weight_dict[k] * v

        total_loss = sum([l for l in loss_dict.values()])

        total_loss.backward()
        optimizer.step()

        log_dict = {**{'loss_' + loss_name: loss_value.item() for loss_name, loss_value in loss_dict.items()},
                    **{'loss': total_loss.item()}
                    }

        for cb in callbacks:
            cb.on_step_fi(log_dict, model, epoch, iteration=total_iter, N=N)

    return callbacks


