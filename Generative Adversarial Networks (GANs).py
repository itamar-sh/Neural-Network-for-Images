import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from collections import namedtuple


# !pip install wandb
import wandb
wandb.login()


transform = transforms.Compose([transforms.ToTensor()])
batch_size = 256
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Function for find test loss
def record_test_loss(net):
    total_loss = 0.0
    i = 1
    # check_images_net(net)
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item() * inputs.size(0)
            i += 1

    num_of_images = inputs.shape[0] * (i)  # batch size * num of batches
    average_loss = total_loss / num_of_images
    print(f'test: num_of_images: {num_of_images}, total loss: {total_loss}, test loss: {average_loss}')
    print(f"total loss: {total_loss}, num of batches: {i}")
    print('Average test set loss per batch: {:.4f}'.format(average_loss))
    print("Input")
    fig, axs = plt.subplots(1, inputs.shape[0], figsize=(10, 10))
    for j in range(inputs.shape[0]):
        vals = inputs[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()
    print("Output")
    fig, axs = plt.subplots(1, inputs.shape[0], figsize=(10, 10))
    for j in range(outputs.shape[0]):
        vals = outputs[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()
    return average_loss


# Function to train the model, and evaluation his permoances while training.
def check_nets(nets):
    for j, net in enumerate(nets):
        wandb.init(
        # Set the project where this run will be logged
        project="Encode Decoder - CNN - AE",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=net.get_name(),
        # Track hyperparameters and run metadata
        config={
        "learning_rate": 0.01,
        "architecture": "CNN",
        "dataset": "CIFAR-50",
        "epochs": 10,
        })


        for epoch in range(3):  # loop over the dataset multiple times
            sum_of_images = 0
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizers[j].zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizers[j].step()

                # print statistics
                running_loss += loss.item() * inputs.size(0)
                # record the train and the test loss
                if i % 150 == 149:
                    num_of_images = inputs.shape[0] * (150)  # batch size * num of batches
                    sum_of_images += num_of_images
                    print(f'train: num_of_images: {num_of_images}, total loss: {running_loss}, train loss: {running_loss / (num_of_images)}, sum_of_images_until_now: {sum_of_images}')
                    cur_test_loss = record_test_loss(net)
                    wandb.log({"train_loss": running_loss / (num_of_images), "test_loss": cur_test_loss})
                    print(f'train epoch: [{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / (num_of_images):.5f} test loss: {cur_test_loss :.3f}')
                    running_loss = 0.0

        wandb.finish()
        print('Finished Training')


# DCGAN architecture
class Generator(nn.Module):
    def __init__(self, input_size=100, num_channels=256):
        super(Generator, self).__init__()

        self.fc = nn.Linear(input_size, num_channels * 4 * 4)
        # Transpose convolutions to increase spatial resolution and reduce number of channels
        self.conv1 = nn.ConvTranspose2d(num_channels, num_channels//2, 3, stride=2, output_padding=1)  # 4 to 10
        self.conv2 = nn.ConvTranspose2d(num_channels//2, num_channels//4, 3, stride=1)  # 10 to 12
        self.conv3 = nn.ConvTranspose2d(num_channels//4, num_channels//8, 3, stride=2, output_padding=1)  # 12 to 26
        self.conv4 = nn.ConvTranspose2d(num_channels//8, 1, 3, stride=1)  # 26 to 28
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.fc(input)
        output = output.view(output.size(0), -1, 4, 4) # reshape to shape=(4 ,4, 1024)

        output = self.conv1(output) # shape=(4 ,4, 1024) to: shape=(10, 10, 512)
        output = nn.functional.relu(output)

        output = self.conv2(output)  # shape=(10, 10, 512) to: shape=(12, 12, 256)
        output = nn.functional.relu(output)

        output = self.conv3(output)  # shape=(12, 12, 256) to: shape=(26, 26, 128)
        output = nn.functional.relu(output)

        output = self.conv4(output)  # shape=(26, 26, 128) to: shape=(28, 28, 1)

        # Sigmoid activation to ensure the images produced are limited to the input range of values (e.g., [0,1])
        output = self.sigmoid(output)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=0)  # 28 hw to 26
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=0) # 26 hw to 12
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=0) # 12 hw to 10
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=0) # 10 hw to 4
        self.fc = nn.Linear(128 * 4 * 4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = self.sigmoid(self.fc(x))
        return x


# Firat question - Loss Saturation
def run_gan_training(gan_parameters, trainloader, device, num_epochs=100, num_disc_updates=3):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = gan_parameters.criterion
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    wandb.init(
            # Set the project where this run will be logged
            project="MNIST Gan",
            name=gan_parameters.name,
            # Track hyperparameters and run metadata
            config={
            "learning_rate": 0.0002,
            "architecture": "DCGAN",
            "dataset": "MNIST",
            "epochs": num_epochs,
            })

    try:
        for epoch in range(num_epochs):
            for i, data in enumerate(trainloader, 0):
                # Update discriminator
                for j in range(num_disc_updates):
                    # Train with real images
                    discriminator.zero_grad()
                    real_images = data[0].to(device)
                    real_labels = torch.ones((real_images.size(0),), device=device)
                    pred_real = discriminator(real_images).flatten()
                    loss_real = criterion(pred_real, real_labels)
                    loss_real.backward()

                    # Train with fake images
                    noise = torch.randn(real_images.size(0), 100, device=device)
                    fake_images = generator(noise)
                    fake_labels = torch.zeros((fake_images.size(0),), device=device)
                    pred_fake = discriminator(fake_images).flatten()
                    loss_fake = criterion(pred_fake, fake_labels)
                    loss_fake.backward()

                    # Update discriminator
                    optimizer_d.step()

                # Update generator
                generator.zero_grad()
                noise = torch.randn(real_images.size(0), 100, device=device)
                fake_images = generator(noise)
                labels = gan_parameters.labels(real_images.size(0))
                pred_gen = discriminator(fake_images).flatten()
                loss_gen = gan_parameters.gen_loss_func(criterion, pred_gen, labels)
                loss_gen.backward()
                optimizer_g.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                          % (epoch, num_epochs, i, len(trainloader), (loss_fake+loss_real).item(), loss_gen.item()))
            print(f"epoch: {epoch}")
            _, axs = plt.subplots(1, 10, figsize=(10, 10))
            for j in range(10):
                vals = fake_images[j].cpu().detach().numpy().squeeze()
                axs[j].imshow(vals, cmap='gray')
                axs[j].axis('off')
            plt.show()
    except Exception:
        raise
    finally:
        wandb.finish()


def run_gans_training():
    GAN_PARAMS = namedtuple('GAN', ['name', 'labels_func', 'criterion', 'gen_loss_func'])
    gans = [
        GAN_PARAMS(
            name="Simple Gan-saturation",
            label_func=lambda real_images: torch.zeros((real_images.size(0),), device=device),
            criterion=nn.BCELoss(),
            loss_calc=lambda criterion, pred_gen, labels: 1 - criterion(pred_gen, labels)
        ),
        GAN_PARAMS(
            name="non_saturation",
            label_func=lambda real_images: torch.ones((real_images.size(0),), device=device),
            criterion=nn.BCELoss(),
            loss_calc=lambda criterion, pred_gen, labels: criterion(pred_gen, labels)
        ),
        GAN_PARAMS(
            name="mse_l2",
            label_func=lambda real_images: torch.ones((real_images.size(0),), device=device),
            criterion=nn.MSELoss(),
            loss_calc=lambda criterion, pred_gen, labels: criterion(pred_gen, labels)
        )
    ]
    for gan in gans:
        run_gan_training(gan, trainloader, device)


def save_generator(generator, drive_path='/content/drive/MyDrive/'):
    from google.colab import drive
    drive.mount('/content/drive')
    torch.save(generator.state_dict(), drive_path + 'generator_model.pth')


def load_generator(drive_path='/content/drive/MyDrive/'):
    generator = Generator()
    generator.load_state_dict(torch.load(drive_path + 'generator_model.pth'))
    generator.eval()  # Set the model to evaluation mode


def Q2():
    for data in testloader:
        input_images, _ = data
        break
    generator = generator.to(device)
    input_images = input_images.to(device)
    print("input")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        vals = input_images[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()
    reconstructed_images = []
    input_images = input_images.to(device)


    criterion = nn.MSELoss()

    for i in range(10):
        z = torch.randn(1, 100, requires_grad=True, device=device)
        optimizer = optim.Adam([z], lr=0.01)
        num_steps = 1000
        for step in range(num_steps):
            optimizer.zero_grad()
            generated_image = generator(z)
            loss = criterion(generated_image, input_images[i])
            loss.backward()
            optimizer.step()
        optimized_z = z.detach()
        reconstructed_images.append(generator(optimized_z))

    reconstructed_images = torch.stack(reconstructed_images)
    print("output")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        vals = reconstructed_images[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()


def Q3_Denoise():
    l1_aggeragated_loss_from_original = 0
    l2_aggeragated_loss_from_original = 0
    l1_aggeragated_loss_from_noised = 0
    l2_aggeragated_loss_from_noised = 0
    generator = generator.to(device)
    for data in testloader:
        input_images, _ = data
        break
    # Print the images
    print("Original Images")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        vals = input_images[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()


    noisy_images = []
    for image in input_images:
        noisy_image = image + torch.randn_like(image) * torch.sqrt(torch.tensor(0.1))
        noisy_images.append(noisy_image)
    noisy_images = torch.stack(noisy_images)
    noisy_images = noisy_images.to(device)

    # Print the noisy images
    print("Noisy Images")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        vals = noisy_images[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()

    # Restore the noisy images using the generator
    restored_images_l1 = []
    restored_images_l2 = []
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()

    for i in range(256):
        z_l1 = torch.randn(1, 100, requires_grad=True, device=device)
        z_l2 = torch.randn(1, 100, requires_grad=True, device=device)
        optimizer_l1 = optim.Adam([z_l1], lr=0.01)
        optimizer_l2 = optim.Adam([z_l2], lr=0.01)
        num_steps = 1000
        for step in range(num_steps):
            optimizer_l1.zero_grad()
            optimizer_l2.zero_grad()
            generated_image_l1 = generator(z_l1)
            generated_image_l2 = generator(z_l2)
            loss_l1 = criterion_l1(generated_image_l1.squeeze(), noisy_images[i].squeeze())
            loss_l2 = criterion_l2(generated_image_l2.squeeze(), noisy_images[i].squeeze())
            loss_l1.backward()
            loss_l2.backward()
            optimizer_l1.step()
            optimizer_l2.step()
        optimized_z_l1 = z_l1.detach()
        optimized_z_l2 = z_l2.detach()
        final_generated_image_l1 = generator(optimized_z_l1)
        final_generated_image_l2 = generator(optimized_z_l2)
        restored_images_l1.append(final_generated_image_l1)
        restored_images_l2.append(final_generated_image_l2)
        input_images = input_images.to(device)
        l1_aggeragated_loss_from_original += criterion_l2(final_generated_image_l1.squeeze(), input_images[i].squeeze()).item()
        l2_aggeragated_loss_from_original += criterion_l2(final_generated_image_l2.squeeze(), input_images[i].squeeze()).item()
        l1_aggeragated_loss_from_noised += criterion_l2(final_generated_image_l1.squeeze(), noisy_images[i].squeeze()).item()
        l2_aggeragated_loss_from_noised += criterion_l2(final_generated_image_l2.squeeze(), noisy_images[i].squeeze()).item()

    restored_images_l1 = torch.stack(restored_images_l1)
    restored_images_l2 = torch.stack(restored_images_l2)
    print("Average of loss of Images generated from L1 norm on the original images: ", l1_aggeragated_loss_from_original / 256)
    print("Average of loss of Images generated from L2 norm on the original images: ", l2_aggeragated_loss_from_original / 256)
    print("Average of loss of Images generated from L1 norm on the noised images: ", l1_aggeragated_loss_from_noised / 256)
    print("Average of loss of Images generated from L2 norm on the noised images: ", l2_aggeragated_loss_from_noised / 256)
    # Print the restored images
    print("Restored Images with L1 loss")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        vals = restored_images_l1[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()

    print("Restored Images with L2 loss")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        vals = restored_images_l2[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()


###########################################################


    import random
    l1_aggeragated_loss_from_original = 0
    l2_aggeragated_loss_from_original = 0
    l1_aggeragated_loss_from_noised = 0
    l2_aggeragated_loss_from_noised = 0
    l1_aggeragated_loss_from_original_2 = 0
    l2_aggeragated_loss_from_original_2 = 0
    l1_aggeragated_loss_from_noised_2 = 0
    l2_aggeragated_loss_from_noised_2 = 0
    generator = generator.to(device)
    for data in testloader:
        input_images, _ = data
        break
    # Print the images
    print("Original Images")
    input_images = input_images.to(device)
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        clone_image = input_images[j].clone()
        vals = clone_image.cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()

    x_y_cords = []
    inpainted_images = []
    for image_index, image in enumerate(input_images):
        clone_image = image.clone()
        x_y_cords.append(list())
        for _ in range(3):  # num of holes
            x = random.randint(0, 20)
            y = random.randint(0, 20)
            x_y_cords[image_index].append((x, y))
            clone_image[:, y:y+8, x:x+8] = 0.0
        inpainted_images.append(clone_image)
    inpainted_images = torch.stack(inpainted_images)
    inpainted_images = inpainted_images.to(device)

    # Print the Inpainted images
    print("Inpainted Images")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        clone_image = inpainted_images[j]
        vals = clone_image.cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()

    # Restore the Inpainted images using the generator
    restored_images_l1 = []
    restored_images_l2 = []
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()

    for i in range(10):
        z_l1 = torch.randn(1, 100, requires_grad=True, device=device)
        z_l2 = torch.randn(1, 100, requires_grad=True, device=device)
        optimizer_l1 = optim.Adam([z_l1], lr=0.01)
        optimizer_l2 = optim.Adam([z_l2], lr=0.01)
        num_steps = 1000
        for step in range(num_steps):
            optimizer_l1.zero_grad()
            optimizer_l2.zero_grad()
            generated_image_l1 = generator(z_l1)
            generated_image_l2 = generator(z_l2)

            for box_index in range(3):  # num of holes
                mask = torch.ones_like(generated_image_l1)
                x, y = x_y_cords[i][box_index]
                mask[:, :, y:y+8, x:x+8] = 0.0
                # Apply the mask to the generated images
                generated_image_l1 = generated_image_l1 * mask
                generated_image_l2 = generated_image_l2 * mask

            loss_l1 = criterion_l1(generated_image_l1.squeeze(), inpainted_images[i].squeeze())
            loss_l2 = criterion_l2(generated_image_l2.squeeze(), inpainted_images[i].squeeze())
            loss_l1.backward()
            loss_l2.backward()
            optimizer_l1.step()
            optimizer_l2.step()
        optimized_z_l1 = z_l1.detach()
        optimized_z_l2 = z_l2.detach()
        final_generated_image_l1 = generator(optimized_z_l1)
        final_generated_image_l2 = generator(optimized_z_l2)
        restored_images_l1.append(final_generated_image_l1)
        restored_images_l2.append(final_generated_image_l2)
        input_images = input_images.to(device)
        l1_aggeragated_loss_from_original += criterion_l2(final_generated_image_l1.squeeze(), input_images[i].squeeze()).item()
        l2_aggeragated_loss_from_original += criterion_l2(final_generated_image_l2.squeeze(), input_images[i].squeeze()).item()
        l1_aggeragated_loss_from_noised += criterion_l2(final_generated_image_l1.squeeze(), inpainted_images[i].squeeze()).item()
        l2_aggeragated_loss_from_noised += criterion_l2(final_generated_image_l2.squeeze(), inpainted_images[i].squeeze()).item()
        l1_aggeragated_loss_from_original_2 += criterion_l1(final_generated_image_l1.squeeze(), input_images[i].squeeze()).item()
        l2_aggeragated_loss_from_original_2 += criterion_l1(final_generated_image_l2.squeeze(), input_images[i].squeeze()).item()
        l1_aggeragated_loss_from_noised_2 += criterion_l1(final_generated_image_l1.squeeze(), inpainted_images[i].squeeze()).item()
        l2_aggeragated_loss_from_noised_2 += criterion_l1(final_generated_image_l2.squeeze(), inpainted_images[i].squeeze()).item()

    restored_images_l1 = torch.stack(restored_images_l1)
    restored_images_l2 = torch.stack(restored_images_l2)
    print("Average of L2 loss of Images generated from L1 norm on the original images: ", l1_aggeragated_loss_from_original / 10)
    print("Average of L2 loss of Images generated from L2 norm on the original images: ", l2_aggeragated_loss_from_original / 10)
    print("Average of L2 loss of Images generated from L1 norm on the inpainted images: ", l1_aggeragated_loss_from_noised / 10)
    print("Average of L2 loss of Images generated from L2 norm on the inpainted images: ", l2_aggeragated_loss_from_noised / 10)
    print("Average of L1 loss of Images generated from L1 norm on the original images: ", l1_aggeragated_loss_from_original_2 / 10)
    print("Average of L1 loss of Images generated from L2 norm on the original images: ", l2_aggeragated_loss_from_original_2 / 10)
    print("Average of L1 loss of Images generated from L1 norm on the inpainted images: ", l1_aggeragated_loss_from_noised_2 / 10)
    print("Average of L1 loss of Images generated from L2 norm on the inpainted images: ", l2_aggeragated_loss_from_noised_2 / 10)
    # Print the restored images
    print("Restored Images with L1 loss")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        vals = restored_images_l1[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()

    print("Restored Images with L2 loss")
    fig, axs = plt.subplots(1, 10, figsize=(10, 10))
    for j in range(10):
        vals = restored_images_l2[j].cpu().detach().numpy().squeeze()
        axs[j].imshow(vals, cmap='gray')
        axs[j].axis('off')
    plt.show()