# Sonifying Explanations for Deep Neural Network Predictions on Speech and Music Data

## Table of Contents  
[W1. Audio Identification vs Audio Reconstruction](#extraction)<br/>
[W2. AudioMNIST - Digits Classification](#digits)  
[W3. AudioMNIST - Gender Classification](#gender)  
[W4. Audio Set](#audioset)  

<a name="extraction"/>

# W1. Audio Identification vs Audio Reconstruction
In this section, we compare the quality of audio extracted for Audio Identification and Audio Reconstruction strategies for two data representations: Spectrogram and Melspectrogram.

1. Audio Extraction for a region from 0-6s and frequencies 0-1800 Hz

Listen to the guitar chords

![Audio Identification (Spectrogram/Melspectrogram)](Master_Thesis_Report_Audio_Files/audio_extractor/ident_6s_1800Hz.wav) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Audio Reconstruction (Spectrogram)](Master_Thesis_Report_Audio_Files/audio_extractor/spect_6s_1800Hz.wav) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Audio Reconstruction (Melspectrogram)](Master_Thesis_Report_Audio_Files/audio_extractor/mspect_6s_1800Hz.wav)

2. Audio Extraction for a region from 0-7.5s and frequencies 0-3500 Hz

Observe the quality of gunshots

![Audio Identification (Spectrogram/Melspectrogram)](Master_Thesis_Report_Audio_Files/audio_extractor/iden_7.5s_3500Hz.wav) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Audio Reconstruction (Spectrogram)](Master_Thesis_Report_Audio_Files/audio_extractor/spect_7.5s_3500Hz.wav) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Audio Reconstruction (Melspectrogram)](Master_Thesis_Report_Audio_Files/audio_extractor/melspect_7.5s_3500Hz.wav)

<a name="digits"/>

# W2. AudioMNIST - Digits Classification
In this section, we provide sonified explanations for AudioMNIST digit classification for spectrogram and melspectrogram data representations and test for *interpretability* aspect. 
Each audio explanation reasoning correct classification follows the pattern: audio for digit 0, its explanation, one-second silence, audio for digit 1, its explanation, one-second silence, and so on until the digit 9. To test the interpretability aspect for digit classification, we inspect whether the sonified explanations are audible and enhance a part of the input important for the prediction. 

**Please use headphones, and raise the volume appropriately. Also, please wait for the audio files to load.**

### W2.1 Spectrogram
**Audio explanations reasoning correct classifications**

![LRP](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/lrp.wav)
![Gradient](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/g.wav)
![DeConvNet](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/dnet.wav)
![DTD](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/dtd.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/gcam.wav)
![Integrated Gradients](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/ig.wav)
![Input*Gradient](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/itg.wav)
![LIME](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/LIME.wav)
![Occlusion](Master_Thesis_Report_Audio_Files/digit_spectrogram/result/Occlusion.wav)

**LRP interpretations for mispredictions**
| Actual and Predicted Labels | Original  | LRP Explanation for misprediction |
|:-------------:|:-------------:|:-----------:|
| Actual label: 2 <br/> Predicted label: 3| ![](Master_Thesis_Report_Audio_Files/digit_spectrogram/mispredictions/act2.wav) | ![](Master_Thesis_Report_Audio_Files/digit_spectrogram/mispredictions/act2_pred3.wav) |
| Actual label: 3 <br/> Predicted label: 6| ![](Master_Thesis_Report_Audio_Files/digit_spectrogram/mispredictions/act3.wav) | ![](Master_Thesis_Report_Audio_Files/digit_spectrogram/mispredictions/act3_pred6.wav) |
| Actual label: 4 <br/> Predicted label: 1| ![](Master_Thesis_Report_Audio_Files/digit_spectrogram/mispredictions/act4.wav) | ![](Master_Thesis_Report_Audio_Files/digit_spectrogram/mispredictions/act4_pred1.wav) |
| Actual label: 5 <br/> Predicted label: 4| ![](Master_Thesis_Report_Audio_Files/digit_spectrogram/mispredictions/act5.wav) | ![](Master_Thesis_Report_Audio_Files/digit_spectrogram/mispredictions/act5_pred4.wav) |

<hr style="border:2px solid gray">

### W2.2 Melspectrogram
**Audio explanations reasoning correct classifications**

![LRP](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/lrp.wav)
![DTD](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/dtd.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/gcam.wav)
![Gradient](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/g.wav)
![DeConvNet](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/dnet.wav)
![Integrated Gradients](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/ig.wav)
![Input*Gradient](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/itg.wav)
![LIME](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/LIME.wav)
![Occlusion](Master_Thesis_Report_Audio_Files/digit_melspectrogram/result/Occlusion.wav)

**LRP interpretations for mispredictions**
| Actual and Predicted Labels | Original  | LRP Explanation for misprediction |
|:-------------:|:-------------:|:-----------:|
| Actual label: 4 <br/> Predicted label: 1| ![](Master_Thesis_Report_Audio_Files/digit_melspectrogram/mispredictions/act4.wav) | ![](Master_Thesis_Report_Audio_Files/digit_melspectrogram/mispredictions/act4_pred1.wav) |
| Actual label: 6 <br/> Predicted label: 3| ![](Master_Thesis_Report_Audio_Files/digit_melspectrogram/mispredictions/act6.wav) | ![](Master_Thesis_Report_Audio_Files/digit_melspectrogram/mispredictions/act6_pred3.wav) |
| Actual label: 7 <br/> Predicted label: 4| ![](Master_Thesis_Report_Audio_Files/digit_melspectrogram/mispredictions/act7.wav) | ![](Master_Thesis_Report_Audio_Files/digit_melspectrogram/mispredictions/act7_pred4.wav) |

<hr style="border:2px solid gray">

<a name="gender"/>

# W3. AudioMNIST - Gender Classification
We test the interpretability aspect for gender classification depending on whether a pitch difference exists between the audio interpretations of male and female classes.  Each audio file to explain correct classification follows this pattern: the digit uttered by a male speaker, its explanation, one second of silence, the same digit uttered by a female speaker, its explanation, and one second of silence.

### W3.1 Spectrogram
**Audio explanations reasoning correct classifications**

![LIME](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/LIME.wav)
![DTD](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/dtd.wav)
![Gradient](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/g.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/gcam.wav)
![LRP](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/lrp.wav)
![DeConvNet](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/dnet.wav)
![Input*Gradient](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/itg.wav)
![Integrated Gradients](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/ig.wav)
![Occlusion](Master_Thesis_Report_Audio_Files/gender_spectrogram/result/Occlusion.wav)

**LIME interpretations for mispredictions**
| Actual and Predicted Labels | Original  | LIME Explanation for misprediction |
|:-------------:|:-------------:|:-----------:|
| Actual label: Male <br/> Predicted label: Female| ![](Master_Thesis_Report_Audio_Files/gender_spectrogram/mispredictions/act0.wav) | ![](Master_Thesis_Report_Audio_Files/gender_spectrogram/mispredictions/act0_pred1.wav) |

<hr style="border:2px solid gray">

### W3.2 Melspectrogram
**Audio explanations reasoning correct classifications**

![Grad-CAM](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/gcam.wav)
![LRP](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/lrp.wav)
![LIME](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/LIME.wav)
![DTD](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/dtd.wav)
![DeConvNet](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/dnet.wav)
![Occlusion](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/Occlusion.wav)
![Input*Gradient](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/itg.wav)
![Integrated Gradients](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/ig.wav)
![Gradient](Master_Thesis_Report_Audio_Files/gender_melspectrogram/result/g.wav)

**Grad-CAM interpretations for mispredictions**
| Actual and Predicted Labels | Original  | Grad-CAM Explanation for misprediction |
|:-------------:|:-------------:|:-----------:|
| Actual label: Male <br/> Predicted label: Female| ![](Master_Thesis_Report_Audio_Files/gender_melspectrogram/mispredictions/act0.wav) | ![](Master_Thesis_Report_Audio_Files/gender_melspectrogram/mispredictions/act0_pred1.wav) |

<hr style="border:2px solid gray">

<a name="audioset"/>

# W4. Audio Set

In this section, we inspect the audio explanations interpreting the yamnet model behavior trained on a real-world dataset, Audio Set containing 2.1 million audio files extracted from YouTube videos. As a popular baseline model in the audio event classification trained on a vast dataset, with audio explanations we identify whether the model predictions are based on the event of interest or not.

For this dataset, we compare audio explanations for five attribution methods: Gradient, Input*Gradient, DeConvNet, Grad-CAM, and LRP. The examples considered are correctly classified by the model and the relevance scores are taken with respect to the output neuron corresponding to the actual class label. Here, correct classification means the top-5 predicted labels contains the Ground truth label. 

Please tune your volume appropriately. We have indicated caution for a few examples with the **Loudness Warning!** sign.

### W4.1 Attribution Methods Comparision

<hr style="border:2px solid gray">

1. **Ground truth**: Gunshot, gunfire

![Original](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/gunshot.wav)
![LRP](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/gunshot_lrp.wav)
![Gradient](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/gunshot_g.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/gunshot_gcam.wav)
![DeConvNet](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/gunshot_dnet.wav)
![Input*Gradient](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/gunshot_itg.wav)

<hr style="border:2px solid gray">

2. **Ground truth**: Bark

![Original](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/bark.wav)
![LRP](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/bark_lrp.wav)
![Gradient](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/bark_g.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/bark_gcam.wav)
![DeConvNet](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/bark_dnet.wav)
![Input*Gradient](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/bark_itg.wav)

<hr style="border:2px solid gray">

3. **Ground truth**: Guitar

![Original](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/guitar.wav)
![LRP](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/guitar_lrp.wav)
![Gradient](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/guitar_g.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/guitar_gcam.wav)
![Input*Gradient](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/guitar_itg.wav)
![DeConvNet](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/guitar_dnet.wav)

<hr style="border:2px solid gray">

4. **Ground truth**: Drums

Listen to the drums being played at 3-4secs

![Original](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/drums.wav)
![LRP](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/drums_lrp.wav)
![Gradient](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/drums_g.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/drums_gcam.wav)
![Input*Gradient](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/drums_itg.wav)
![DeConvNet](Master_Thesis_Report_Audio_Files/AudioSet/methods_comparision/drums_dnet.wav)

<hr style="border:2px solid gray">

#### LRP vs Grad-CAM Explanations

5. **Ground truth**: Heart sounds, heartbeat (**Increase the volume for this example, preferably use headphones**)

![Original](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/heart14.wav)
![LRP](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/lrp_heart14.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/gcam_heart14.wav)

<hr style="border:2px solid gray">

6. **Ground truth**: Drums

![Original](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/Drum.2.wav)
![LRP](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/lrp_Drum_2.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/gcam_Drum_2.wav)

<hr style="border:2px solid gray">

7. **Ground truth**: Smoke detector, smoke alarm (**Loudness Warning!**)

![Original](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/smoke2.wav)
![LRP](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/lrp_smoke2.wav)
![Grad-CAM](Master_Thesis_Report_Audio_Files/AudioSet/lrp_vs_gcam/gcam_smoke2.wav)

<hr style="border:2px solid gray">

### W4.2 Noteworthy (Cherry-picked) LRP Explanations for correct classifications 
Please pay close attention to the original audio for a particular class and the relevant audio responsible for the model decision in the case of correct classifications.

| Category | Class | Original Input Audio | LRP Audio Explanation  |
|:-------------:|:-------------:|:-----------:|:----------:|
| Musical Instrument | Harp | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Harp.10.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Harp.10_lrp.wav) |
| Musical Instrument | Organ | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Organ.21.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Organ.21_lrp.wav) |
| Musical Instrument | Drums | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Drum.21.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Drum.21_lrp.wav) |
| Musical Instrument | Guitar | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Guitar.17.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Guitar.17_lrp.wav) |
| Music Genre | Blues | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Blues.21.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Blues.21_lrp.wav) |
| Music Genre | Country | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Country.23.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Country.23_lrp.wav) |
| Siren | Fire engine, fire truck (siren) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/firesiren.24.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/firesiren.24_lrp.wav) |
| Gunshot | Gunshot, gunfire | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/gunshot.29.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/gunshot.29_lrp.wav) |
| Whistle | Whistling | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Whistling.19.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/noteworthy_lrp/Whistling.19_lrp.wav) |

<hr style="border:2px solid gray">

### W4.3 Random LRP Explanations for correct classifications 
Please pay close attention to the original audio for a particular class and the relevant audio responsible for the model decision in the case of correct classifications.

| Category | Class | Original  | LRP Explanation  |
|:-------------:|:-------------:|:-----------:|:----------:|
| Musical Instrument | Harp | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Harp.2.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Harp_2_relevant.wav) |
| Musical Instrument | Organ | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Organ.26.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Organ_26_relevant.wav) |
| Musical Instrument | Drums | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Drum.18.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Drum_18_relevant.wav) |
| Musical Instrument | Guitar | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Guitar.8.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Guitar_8_relevant.wav) |
| Music Genre | Blues | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Blues.15.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Blues_15_relevant.wav) |
| Music Genre | Country | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Country.8.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Country_8_relevant.wav) |
| Alarm | Smoke detector, smoke alarm | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/smoke_detector.28.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/smoke_detector_28_relevant.wav) |
| Siren | Ambulance (Siren) <br/>(**Loudness Warning!**) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Ambulance_(siren).17.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/random_lrp/Ambulance_(siren)_17_relevant.wav) |

<hr style="border:2px solid gray">

### W4.4 Analysing LRP Explanations for mispredictions
For model mispredictions, we provide audio interpretations for the predicted class labels portraying the reason for mispredictions. For a few examples, we have also observed the explanation for the ground truth label.

1. **Ground truth**: Alarm

   **Top-K Predicted labels**: Speech, Buzzer, Smoke detector smoke alarm, Beep bleep, Music

 Original | Intepretation for Speech | Intepretation for Alarm
|:-----------:|:----------:|:----------:|
![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Alarm.20.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Alarm_20_pred.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Alarm_20_act.wav) |

<hr style="border:2px solid gray">

2. **Ground truth**: Bark

   **Top-K Predicted labels**: Speech, Animal, Domestic animals/ pets, Dog, Outside/ rural or natural

 Original | Intepretation for Speech | Intepretation for Bark
|:-----------:|:----------:|:----------:|
![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Bark.2.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Bark_2_pred.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Bark_2_act.wav) |

<hr style="border:2px solid gray">

3. **Ground truth**: Drums

   **Top-K Predicted labels**: Explosion, Boom, Artillery fire, Eruption, Gunshot/ gunfire

 Original | Intepretation for Explosion
|:-----------:|:----------:|
![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Drum.16.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Drum_16_pred.wav) |

<hr style="border:2px solid gray">

4. **Ground truth**: Alarm

   **Top-K Predicted labels**: Flute, Wind/ woodwind instrument, Music, Animal, Dog

 Original | Intepretation for Flute
|:-----------:|:----------:|
![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Alarm.0.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Alarm_0_pred.wav) 

<hr style="border:2px solid gray">

5. **Ground truth**: Gunshot/ gunfire

   **Top-K Predicted labels**: Music, Vehicle, Rail transport, Train, Railroad car/ train wagon

 Original | Intepretation for Music
|:-----------:|:----------:|
![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Gunshot.21.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Gunshot.21_pred.wav) 

<hr style="border:2px solid gray">

6. **Ground truth**: Country

   **Top-K Predicted labels**: Music, Musical instrument, Bowed string instrument, Banjo, Violin/ fiddle

 Original | Intepretation for Music
|:-----------:|:----------:|
![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Country.15.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Country_15_pred.wav) 


<hr style="border:2px solid gray">

7. **Ground truth**: Guitar

   **Top-K Predicted labels**: Music, Slosh, Splash/, splatter, Liquid, Gurgling

 Original | Intepretation for Music
|:-----------:|:----------:|
![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Guitar.29.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Guitar_29_pred.wav) 


<hr style="border:2px solid gray">

8. **Ground truth**: Whistling

   **Top-K Predicted labels**: Animal, Wild animals, Bird, Speech, Pigeon/ dove

 Original | Intepretation for Animal
|:-----------:|:----------:|
![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Whistling.14.wav) | ![](Master_Thesis_Report_Audio_Files/AudioSet/misclassifications_lrp/Whistling_14_pred.wav) 
