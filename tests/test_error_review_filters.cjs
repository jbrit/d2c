const assert = require('node:assert/strict');
const {matchingResults, selectRecords, csvCell} = require('../scripts/review_errors.js');
const record = {id:10, filename:'0010.jpg',filepath:'/dataset/0010.jpg',split:'validation',casualty:'casualty_1',
  issues:['uncertain'],cropCaution:false,results:[
    ['a','arm_left','no_injury','wound',.9],
    ['a','leg_left','wound','no_injury',.8],
    ['b','leg_left','wound','wound',.7],
    ['majority','leg_left','wound','no_injury',null]
  ]};
assert.equal(matchingResults(record,{model:'a',limb:'arm_left',truth:'wound',errors:true}).length,0,
  'Limb and supplied-class filters must match the same comparison');
assert.equal(matchingResults(record,{model:'b',truth:'wound',prediction:'no_injury',errors:true}).length,0,
  'Predictions cannot be borrowed from another model');
assert.equal(matchingResults(record,{model:'a',limb:'leg_left',truth:'wound',prediction:'no_injury',errors:true}).length,1);
assert.equal(matchingResults(record,{model:'learned',errors:true}).length,2,'Learned-only excludes majority');
assert.equal(matchingResults(record,{model:'b',errors:false}).length,1,'Correct controls remain inspectable');
assert.equal(matchingResults(record,{split:'train',errors:true}).length,0,'Splits remain separate');
assert.equal(matchingResults(record,{issue:'likely_mismatch',errors:true}).length,0);
assert.equal(matchingResults(record,{issue:'uncertain',search:'0010.JPG',errors:true}).length,3);
assert.equal(selectRecords([record],{errors:true}).length,1,'Multiple errors do not duplicate an image');
assert.equal(csvCell('a,"b"'),'"a,""b"""');
console.log('10 error-review filter checks passed');
