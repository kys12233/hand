import os


save_list = []
image_paths = 'F:\Data_Set\HandwrittenNumeralRecognition\mnist_test'

save_label = r'F:\Vscode_python_programs\HandwrittenNumeralRecognition\data_to\test_label.txt'

for i in os.listdir(image_paths):
    for j in os.listdir(image_paths + '\\' +i):
        save_list_one = []
        save_list_one.append(image_paths + '\\' +i + '\\' + j)
        save_list_one.append(i)

        save_list.append(save_list_one)


with open(save_label,"w") as file:
    for i in range(len(save_list)):
        file.write(save_list[i][0] + ',')
        file.write(save_list[i][1] + '\n')
    file.close()
    exit()



