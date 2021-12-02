import pickle
import matplotlib.pyplot as plt


def analysis_train(outputs, out_fname='analysis_train.png'):
    counter = 1
    # Plotting reconstructions
    # for epochs = [1, 5, 10, 50, 100]
    epochs_list = [1, 5, 10]
    # epochs_list = [1, 5, 10, 50, 100]
    # Iterating over specified epochs
    for val in epochs_list:
        # Extracting recorded information
        temp = outputs[val]['out'].detach().numpy()
        title_text = f"Epoch = {val}"
        # Plotting first five images of the last batch
        for idx in range(5):
            plt.subplot(7, 5, counter)
            plt.title(title_text)
            plt.imshow(temp[idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
            # Incrementing the subplot counter
            counter += 1
    # Plotting original images
    # Iterating over first five
    # images of the last batch
    for idx in range(5):
        # Obtaining image from the dictionary
        val = outputs[10]['img']
        # Plotting image
        plt.subplot(7, 5, counter)
        plt.imshow(val[idx].reshape(28, 28), cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        # Incrementing subplot counter
        counter += 1
    plt.tight_layout()
    plt.show()
    plt.savefig(out_fname)
    plt.cla()
    plt.clf()


def analysis_test(model, test_loader, out_fname='analysis_test.png'):
    plt.rc('font', size=4)
    # Dictionary that will store the different
    # images and outputs for various epochs
    outputs = {}
    # Extracting the last batch from the test
    # dataset
    img, _ = list(test_loader)[-1]
    # Reshaping into 1d vector
    img = img.reshape(-1, 28 * 28)
    # Generating output for the obtained
    # batch
    out = model(img)
    # Storing information in dictionary
    outputs['img'] = img
    outputs['out'] = out
    # Plotting reconstructed images
    # Initializing subplot counter
    counter = 1
    val = outputs['out'].detach().numpy()
    # Plotting first 10 images of the batch
    for idx in range(10):
        plt.subplot(2, 10, counter)
        plt.title("Reconstructed")
        plt.imshow(val[idx].reshape(28, 28), cmap='gray')
        plt.axis('off')
        # Incrementing subplot counter
        counter += 1
    # Plotting original images
    # Plotting first 10 images
    for idx in range(10):
        val = outputs['img']
        plt.subplot(2, 10, counter)
        plt.imshow(val[idx].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        # Incrementing subplot counter
        counter += 1
    plt.tight_layout()
    plt.show()
    plt.savefig(out_fname)
    plt.cla()
    plt.clf()
