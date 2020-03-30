# SideSolver
I am making a puzzle scanner and solver to learn more about OpenCV, Tensorflow and Python. Because coding is fun. 


### Some progress:


21 March 2020 - *Original*:  
Started with the project. This is the original in grayscale

![21 March 2020 - *Original*: Started with the project. This is the original in grayscale](progress/original.png)  

22 March 2020 - *Thresholding*:  
Played around with different thresholding methods and found adaptive gaussian to indeed work the best. (Makes everything either black or white

![22 March 2020 - *Thresholding*: Played around with different thresholding methods and found adaptive gaussian to indeed work the best. (Makes everything either black or white)](progress/thresholded.png)  

30 March 2020 - *Floodfilling and blob detection*  
I found OpenCV's blob function to not quite work the way I wanted (and I could get it to work immediately) so I wrote my own (its really slow and cringy and but it works ;)

![30 March 2020 - *Floodfilling and blob detection*: I found OpenCV's blob function to not quite work the way I wanted (and slightly difficult to use) so I wrote my own (its really slow and cringy and but it works :sunglasses: )](progress/grid.png)  

### To do:
- Add pipeline to check style and unit tests. 
- Code coverage could be cool thing to add 

- Plot grid to new image (this way we can throw away everything outside of the grid, making it easier to determine position of digits).
- Determine locations of digits
- Identify digits
- Map final product to array and solve

- Experiment with Deep Belief Network

- Optimize and Neaten 

## Dudes I basically plagiarized/ got inspiration from (gotta start somewhere):
- http://sudokugrab.blogspot.com/2009/07/how-does-it-all-work.html

- https://aishack.in/tutorials/sudoku-grabber-opencv-detection


### Constructive critism welcome. #BeLekker