import type { MPM2D } from './mpm2d';

export type Scene2D = (sim: MPM2D) => void;

export const SCENES_2D: Record<string, Scene2D> = {
  'Drop jelly': (sim) => {
    sim.addBox(0.28, 0.55, 0.2, 0.2, 0, 0, 0);
    sim.addBox(0.62, 0.78, 0.16, 0.12, 0, -1.5, 1);
    sim.addDisk(0.72, 0.38, 0.08, -0.5, 0, 2);
  },
  'Throw at wall': (sim) => {
    sim.addBox(0.25, 0.35, 0.18, 0.18, 4, 1.5, 0);
    sim.addDisk(0.25, 0.75, 0.07, 3, 0, 3);
  },
  'Jelly stack': (sim) => {
    sim.addBox(0.5, 0.13, 0.4, 0.12, 0, 0, 1);
    sim.addBox(0.5, 0.33, 0.24, 0.16, 0, 0, 0);
    sim.addDisk(0.5, 0.75, 0.08, 0, -2, 2);
  },
};
