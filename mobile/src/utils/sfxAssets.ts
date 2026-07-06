/**
 * Impact SFX asset map.
 *
 * Maps action codes (from combo_dictionary.json) to arrays of require()'d
 * MP3 assets. Multiple variants per strike type for natural variation.
 */

/* eslint-disable @typescript-eslint/no-var-requires */

const jab1 = require('../../assets/sfx/impact/jab_1.mp3');
const jab2 = require('../../assets/sfx/impact/jab_2.mp3');
const jab3 = require('../../assets/sfx/impact/jab_3.mp3');

const cross1 = require('../../assets/sfx/impact/cross_1.mp3');
const cross2 = require('../../assets/sfx/impact/cross_2.mp3');
const cross3 = require('../../assets/sfx/impact/cross_3.mp3');

const hook1 = require('../../assets/sfx/impact/hook_1.mp3');
const hook2 = require('../../assets/sfx/impact/hook_2.mp3');
const hook3 = require('../../assets/sfx/impact/hook_3.mp3');

const uppercut1 = require('../../assets/sfx/impact/uppercut_1.mp3');
const uppercut2 = require('../../assets/sfx/impact/uppercut_2.mp3');
const uppercut3 = require('../../assets/sfx/impact/uppercut_3.mp3');

const body1 = require('../../assets/sfx/impact/body_1.mp3');
const body2 = require('../../assets/sfx/impact/body_2.mp3');
const body3 = require('../../assets/sfx/impact/body_3.mp3');

const slip = require('../../assets/sfx/impact/slip.mp3');
const duck = require('../../assets/sfx/impact/duck.mp3');
const weave = require('../../assets/sfx/impact/weave.mp3');
const back = require('../../assets/sfx/impact/back.mp3');

const chain1 = require('../../assets/sfx/impact/chain_1.mp3');
const chain2 = require('../../assets/sfx/impact/chain_2.mp3');

// Action code → variant assets (strike types have 3 variants, defense has 1)
export const IMPACT_ASSETS: Record<string, any[]> = {
  // Strikes
  J:  [jab1, jab2, jab3],
  C:  [cross1, cross2, cross3],
  LH: [hook1, hook2, hook3],
  RH: [hook1, hook2, hook3],
  LU: [uppercut1, uppercut2, uppercut3],
  RU: [uppercut1, uppercut2, uppercut3],
  LB: [body1, body2, body3],
  RB: [body1, body2, body3],

  // Defense
  SL: [slip],
  DK: [duck],
  WV: [weave],
  BS: [back],
};

// Chain rattle — played after combos ending with hooks/uppercuts
export const CHAIN_ASSETS = [chain1, chain2];

// Actions that trigger chain rattle after them
export const CHAIN_TRIGGER_ACTIONS = new Set(['LH', 'RH', 'LU', 'RU']);
