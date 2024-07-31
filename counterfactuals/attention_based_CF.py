import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeries
from data_processing.time_series_processing import sliding_window_3d
import time

def entropy(predict_proba):
    predict_proba = predict_proba[np.nonzero(predict_proba)]
    return -np.sum(predict_proba * np.log2(predict_proba))

def native_guide_retrieval(query, target_label, distance, n_neighbors, X_train, y_train):
    df = pd.DataFrame(y_train, columns=['label'])
    df.index.name = 'index'

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
    knn.fit(X_train[df[df['label'] == target_label].index.values])

    _, ind = knn.kneighbors(query.reshape(1, query.shape[0], query.shape[1]), return_distance=True)
    return df[df['label'] == target_label].index[ind[0][0]]

def generate_counterfactuals(X_test, y_test, model, X_train, y_train, window_size, stride):
    ts_length = X_train.shape[2]
    cfs = []
    target_probas = []
    times = []
    suc_indexes = []
    num_dim_changed = []

    for i in range(len(X_test[:260])):
        if y_test[i] == 2:
            continue
        print("Processing index", i)
        start_time = time.time()

        subsequences = sliding_window_3d(X_test[i], window_size, stride)
        padded_subsequences = np.pad(subsequences, ((0, 0), (0, 0), (0, ts_length - subsequences.shape[2])),
                                    mode='constant')
        predict_proba = model.predict(padded_subsequences)
        pred = model.predict(X_test[i].reshape(1, X_test[i].shape[0], X_test[i].shape[1]))

        entropies = [entropy(p) for p in predict_proba]
        indices = np.argsort(entropies)[:10]
        min_entropy_index = np.argmin(entropies)

        if np.argmax(pred) != y_test[i]:
            target = y_test[i]
        else:
            target = target_(model, X_test[i])

        idx = native_guide_retrieval(X_test[i], target, 'dtw', 1, X_train, y_train)
        nun = X_train[idx.item()]
        cf = X_test[i].copy()

        for k, index in enumerate(indices, start=1):
            start = index * stride
            end = start + window_size
            cf[0, start:end] = nun[0, start:end] + 50
            cf = cf.reshape(1, cf.shape[0], cf.shape[1])

            if np.argmax(model.predict(cf)) == target:
                print("Counterfactual generation successful for index", i)
                suc_indexes.append(i)
                target_probas.append(model.predict(cf)[0][target])
                times.append(time.time() - start_time)
                num_dim_changed.append(k)
                cfs.append(cf)
                break
            else:
                cf = cf.reshape(cf.shape[1], cf.shape[2])
                
    return cfs, times, target_probas, num_dim_changed, suc_indexes

# def generate_counterfactuals(X_test, y_test, model, X_train, y_train, window_size, stride, max_value=1000):
#     ts_length = X_train.shape[2]
#     cfs = []
#     target_probas = []
#     times = []
#     suc_indexes = []
#     num_dim_changed = []

#     for i in range(len(X_test[:20])):
#         if y_test[i] == 2:
#             continue
#         print("Processing index", i)
#         start_time = time.time()

#         subsequences = sliding_window_3d(X_test[i], window_size, stride)
#         padded_subsequences = np.pad(subsequences, ((0, 0), (0, 0), (0, ts_length - subsequences.shape[2])),
#                                      mode='constant')
#         predict_proba = model.predict(padded_subsequences)
#         pred = model.predict(X_test[i].reshape(1, X_test[i].shape[0], X_test[i].shape[1]))

#         entropies = [entropy(p) for p in predict_proba]
#         indices = np.argsort(entropies)[:10]

#         if np.argmax(pred) != y_test[i]:
#             target = y_test[i]
#         else:
#             target = target_(model, X_test[i])

#         cf = X_test[i].copy()

#         # Iterate over each subsequence window and try values from 0 to max_value
#         for k, index in enumerate(indices, start=1):
#             start = index * stride
#             end = start + window_size

#             # Try replacing the subsequence with values from 0 to max_value
#             for value in range(0, max_value + 1, 10):
#                 cf[0, start:end] = value
#                 cf = cf.reshape(1, cf.shape[0], cf.shape[1])

#                 if np.argmax(model.predict(cf)) == target:
#                     print(f"Counterfactual generation successful for index {i} with value {value} at step {k}.")
#                     suc_indexes.append(i)
#                     target_probas.append(model.predict(cf)[0][target])
#                     times.append(time.time() - start_time)
#                     num_dim_changed.append(k)
#                     cfs.append(cf)
#                     break
#                 else:
#                     cf = cf.reshape(cf.shape[1], cf.shape[2])
#             else:
#                 continue
#             break

#     return cfs, times, target_probas, num_dim_changed, suc_indexes



def target_(model, instance):
    return np.argsort(model.predict(instance.reshape(1, instance.shape[0], instance.shape[1])))[0][-2]


def plot_original_vs_counterfactuals(X_test, y_test, cfs, model, suc_indexes, save_path="results/plots/"):
    cfs_array = []
    for cf in cfs:
        cfs_reshaped = cf.reshape(1, 3, 10)
        cfs_array.append(cfs_reshaped)
    cfs_array = np.concatenate(cfs_array)
    print(cfs_array.shape)
    y_cfs = model.predict(cfs_array)

    # n = ['steps', 'heart', 'glucose']
    # for l, m in enumerate(suc_indexes):
    #     t = np.linspace(0, 15*X_test.shape[2], num=X_test.shape[2])
    #     data = X_test[m]
    #     y_ori = y_test[m]
    #     data_cf = cfs[l]
        
    #     y_P = np.argmax(model.predict(data.reshape(1, 3, 10)))
    #     y_cf = np.argmax(model.predict(data_cf.reshape(1, 3, 10)))

    #     # Create a figure with two subplots
    #     fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    #     # Plot original data
    #     for i in range(len(n)):
    #         axs[0].plot(t, data[i, :], label=f"{n[i]}")
    #     axs[0].set_xlabel("Time (m)")
    #     axs[0].set_ylabel("Value")
    #     axs[0].legend()
    #     axs[0].set_title(f"Original Data (True: {y_ori}, Pred: {y_P})")

    #     # Plot counterfactual data
    #     for i in range(len(n)):
    #         axs[1].plot(t, data_cf[0][i, :], label=f"{n[i]}")
    #     axs[1].set_xlabel("Time (m)")
    #     axs[1].set_ylabel("Value")
    #     axs[1].legend()
    #     axs[1].set_title(f"Counterfactual Data (True: {y_ori}, CF Pred: {y_cf})")

    #     # Adjust layout and save the plot
    #     plt.tight_layout()
    #     plt.savefig(f"{save_path}plot_{m}.png")
    #     plt.close(fig)



    n = ['steps', 'heart', 'glucose']
    for l, m in enumerate(suc_indexes):
        t = np.linspace(0, 15 * X_test.shape[2], num=X_test.shape[2])
        data = X_test[m]
        y_ori = y_test[m]
        data_cf = cfs[l]
        
        y_P = np.argmax(model.predict(data.reshape(1, 3, 10)))
        y_cf = np.argmax(model.predict(data_cf.reshape(1, 3, 10)))

        # Create a figure with two subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Plot original data
        for i in range(len(n)):
            axs[0].plot(t, data[i, :], label=f"{n[i]}")
            if n[i] == 'steps':
                for x, y in zip(t, data[i, :]):
                    axs[0].text(x, y, f"{y}", fontsize=8, alpha=0.7)
        axs[0].set_xlabel("Time (m)")
        axs[0].set_ylabel("Value")
        axs[0].legend()
        axs[0].set_title(f"Original Data (True: {y_ori}, Pred: {y_P})")

        # Plot counterfactual data
        for i in range(len(n)):
            axs[1].plot(t, data_cf[0][i, :], label=f"{n[i]}")
            if n[i] == 'steps':
                for x, y in zip(t, data_cf[0][i, :]):
                    axs[1].text(x, y, f"{y}", fontsize=8, alpha=0.7)
        axs[1].set_xlabel("Time (m)")
        axs[1].set_ylabel("Value")
        axs[1].legend()
        axs[1].set_title(f"Counterfactual Data (True: {y_ori}, CF Pred: {y_cf})")

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(f"{save_path}plot_{m}.png")
        plt.close(fig)
