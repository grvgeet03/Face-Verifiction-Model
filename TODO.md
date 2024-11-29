**These are the steps you need to setup to get app functioning on your device**

Create a folder for the project and clone the repository there.
Create the virtual environment and download the dependencies from requirements.txt
Run the Jupyter Notebook, modify the code where required and execute the code.
You only have to create some of the directories in the steps ahead like application_data folder, pasting the model in the app folder. The rest will be created them selves
Then follow the steps given below.


1. Setup app folder
2. Install Kivy
3. Setup validation folder (paste images from positive in this folder)
4. Create custom layer module (layers.py)
5. Bring over keras model (The saved model)

6. Import dependencies for Kivy
7. Build Layout
8. Build update function
9. Bring over preprocessing function (From Jupyter)

10. Bring verification function 
11. Update  verificaiton function  ro handle new paths and save current frame
12. Update verification function to set verified text
13. Link the verificaiton function to the button
14. Setup Logger