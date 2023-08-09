# Ray Tracing implementation in Python

Most of the code is Linear Algebra calculations combined with Physics to approximate the colors of a scene, external libraries were used only for faster calculations.

Scenes can be manually created using a text document that is later parsed in the code to indentify relevant objects and interpret the scene.

The code is capable of rendering cubes, spheres and planes, it utilizes the Phong model for the material properties (calculating Ambient, Diffuse and Specular to estimate the colors of the object).

The code is also capable of rendering within a chosen recursion depth (amount of times a ray can bounce until is stops) and rendering transparent and semi-transparent objects.

The code also implements Soft Shadows approach which is computationally heavy but more realistic (a grid of rays is shot for each light source to estimate amount of light recieved from a non-zero volume light source to a given point).

 <ins>Render output example for original2.txt: </ins>

![original2](https://github.com/ItsAiZEN/Ray-Tracing-implementation-in-Python/assets/64685062/992ac7e6-2f15-42b9-955b-46a59ab97575)
