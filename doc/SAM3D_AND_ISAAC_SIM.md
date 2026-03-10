# SAM 3D output format and path to Isaac Sim

## What SAM 3D Objects outputs

From the [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) repo:

- **Primary output: 3D Gaussian Splatting in PLY format.**  
  The demo and API use `output["gs"].save_ply("splat.ply")`. So each object is saved as a **PLY file** that stores **Gaussian splats** (positions, covariance, opacity, spherical harmonics for color), not a triangle mesh.

- **No mesh/OBJ in the default API.**  
  The public inference code returns a dict with at least `"gs"` (the Gaussian splat representation). The repo does not expose a built-in export to mesh (OBJ/FBX) or USD in the minimal demo; internal representations may include geometry used for rendering, but the documented export is PLY (splat).

So in this pipeline, **one masked object → one `.ply` file (Gaussian splat).**

---

## Next steps after SAM 3D (before simulating in Isaac Sim)

Isaac Sim (and NVIDIA Isaac Lab) typically expect **USD scenes** and **mesh-based or rigid-body assets** (collision, physics, articulation). Gaussian splat PLYs are view-dependent renderable representations, not meshes, so you usually need a conversion step.

1. **Convert splat PLY → mesh (optional but typical for sim)**  
   - There is no standard “splat → mesh” in the SAM 3D repo. Options:
     - Use a **point-cloud / splat → mesh** tool (e.g. Poisson reconstruction from splat centers, or a converter that turns splats into a mesh).
     - Some tools (e.g. [SplatTransform](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/splat-transform/)) convert between splat formats; going to a **sim-ready mesh** often requires an extra step (export points → surface reconstruction → OBJ/FBX/USD).
   - Result you want for sim: **mesh format (OBJ, FBX, or USD)** with consistent scale and orientation.

2. **Import into Isaac Sim / Isaac Lab**  
   - Use **USD** as the scene format. Isaac Lab’s [MeshConverter](https://docs.robotsfan.com/isaaclab_official/v2.0.1/_modules/isaaclab/sim/converters/mesh_converter.html) supports **OBJ, STL, FBX** → USD. So:
     - If you have **OBJ/STL/FBX** (from step 1), convert them to USD and place them in your scene.
     - If you stay with **PLY (splat)** only, Isaac Sim does not natively support Gaussian splats for physics; you’d need to either convert to mesh first or use splats only for visualization in a different pipeline.

3. **Scene setup for manipulation (e.g. opening drawers)**  
   - **Rigid bodies**: assign to static and dynamic objects.  
   - **Articulation**: for drawers, doors, etc., define joints (prismatic for drawers, revolute for doors) in USD or via Isaac Sim’s articulation APIs.  
   - **Collision**: ensure collision meshes (or simplified collision shapes) are set so the robot can interact with the objects.  
   - **Scaling and units**: match your SAM 3D / reconstruction scale to Isaac Sim’s world units (e.g. meters).

4. **Robot and task**  
   - Add the robot (e.g. from Isaac Lab / Omniverse) and attach controllers.  
   - Train or script policies (RL, imitation, or scripted) for tasks like “open drawer” using the imported 3D objects and articulations.

**Summary:**  
**SAM 3D output = PLY (Gaussian splat) per object.** For Isaac Sim, convert PLY → mesh (OBJ/FBX) if needed, then OBJ/FBX → USD with MeshConverter, then add physics/articulation and the robot.
