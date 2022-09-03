## Using Photong
When you first open the page, you may see a few spinners and loading texts. Please wait patiently until they disappear. Then you should see a file uploader.

![Screenshot from a webpage with the heading "Photong", a link to this tutorial, and a file uploader that accepts PNG, JPG, and JPEG images.](imgs/01-main.png)

Click on the uploader. You will be prompted to select an image. Note that only images with extensions `.png`, `.jpg` and `.jpeg` images are supported.

After the image is uploaded successfully, the app will start running a sequence of steps. Eventually, it will tell you if it thinks the image is happy or sad, exciting or peaceful.

![Screenshot from a webpage with the heading "Photong", a link to this tutorial, a file called "sad.png" uploaded, and two lines of text that says "I think it's a sad image." and "I think it's a peaceful image."](imgs/02-information.png)

The generation process may take a while, during which some models may need to be downloaded and loaded in the app.

If the process succeeds, you will see a new section called "Result". The app will show you the original image you uploaded and a green box that says "Here is your melody!". Below this box, you should be able to see a music player that can play the melody for you. Click on the play triangle button to listen to the generated melody.

![Screenshot from a webpage with the heading "Result", a sad image, a green success box that says "Here is your melody!", and a music player.](imgs/03-result.png)

To save the generated melody, you can click on the three dots on the right of the music player and select "Download". The audio file will automatically be saved to your computer.

![Screenshot from a webpage showing a menu of the music player with the "Download" option active.](imgs/04-download.png)

Alternatively, you can click on the "Download MIDI" button below the music player and save the MIDI file to your computer. A MIDI file can be opened by any musical notation or production software, such as [MuseScore](https://musescore.org/) or any DAW (Digital Audio Workstation) software.

If you run into a problem, please [open an issue](https://github.com/photong-ml/photong-streamlit/issues/new).
