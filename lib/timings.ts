export interface TrialWindow {
  slot: 1 | 2 | 3;
  rest: [number, number];
  stimulus: [number, number];
  imagination: [number, number];
  keep: [number, number];
}

export const WORD_LEVEL_WINDOWS: TrialWindow[] = [
  {
    slot: 1,
    rest: [0.0, 5.0],
    stimulus: [5.0, 10.0],
    imagination: [10.0, 16.0],
    keep: [10.5, 16.0]
  },
  {
    slot: 2,
    rest: [16.0, 21.0],
    stimulus: [21.0, 26.0],
    imagination: [26.0, 32.0],
    keep: [26.5, 32.0]
  },
  {
    slot: 3,
    rest: [32.0, 37.0],
    stimulus: [37.0, 42.0],
    imagination: [42.0, 48.0],
    keep: [42.5, 48.0]
  }
];

export const SENTENCE_TRIAL_SECONDS = 48;
