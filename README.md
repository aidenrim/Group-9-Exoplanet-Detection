Group 9 Exoplanet Detection
Members: Ayush Adhikari, Nikhil Khot, Arha Yalamanchili, Aiden Rim 


AI Technique/Method used: Resnet 18
Data from: MAST Queries from Astroquery

Exoplanet detection is an ongoing problem within the field of astronomy. Exoplanets are planets not located within our solar system - most commonly orbiting another star. They are among the top candidates for finding life outside of Earth, motivating their study. Beyond that, each confirmed exoplanet challenges and/or strengthens our knowledge of planetary formation as the unique properties of each star system / exoplanet(s) pair often greatly vary from the few planets we have been able to directly image within our own system. 

Due to their distance, exoplanets are hard to directly image. Instead, missions such as Kepler and TESS (Transiting Exoplanet Survey Satellite) collect data on the change in the brightness of stars over a large period of time. The dips in brightness are then analyzed to determine whether they are caused by the existence of an exoplanet in orbit, or by some other mechanism. While discoveries are still confirmed by hand, AI techniques are commonly used to narrow down the large volume of data and identify likely candidates for humans to verify. 

Previous approaches to this topic include AstroNet, an open source project by Google Research that used manually vetted data to train a convolutional neural network that would predict the probability that a given dip in a light curve is caused by an exoplanet in transit. AstroNet also used methods such as manually occluding sections of given light curves and checking the change in output probability to determine which sections of the curve were being used by the model to make classifications. 

Another more recent project is ExoMiner, which seeks to validate potential exoplanet candidates through a deep neural network built with the data collected by Kepler and TESS. Their goal is to classify transit signals, create catalogs of Threshold Crossing Events used to identify planetary candidates, then figure out what corresponds to exoplanets. 

Our goal is to flag candidates of interest from the ongoing TESS mission as likely/unlikely exoplanets. We plan on utilizing photometric data from both the Kepler and TESS missions to build our AI. Though we have not yet decided what AI techniques to use, we plan to attempt a different approach to the problem than previously explored.
