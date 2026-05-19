/**
 * NOISE ELIMINATION BACKTEST
 * Goal: Find rules to safely eliminate as many numbers as possible 
 * WITHOUT killing the winning numbers.
 */
const https = require('https');
function fetchData() { return new Promise((res, rej) => { https.get('https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl', r => { let d='';r.on('data',c=>d+=c);r.on('end',()=>{const dr=[];for(const l of d.trim().split('\n')){if(!l)continue;const o=JSON.parse(l);if(o.result?.length>=6)dr.push(o.result.slice(0,6).map(Number).sort((a,b)=>a-b));}res(dr);});r.on('error',rej);}); }); }

function runElimination(allData) {
    let totalKilled = 0;
    let safeKilled = 0; // Killed numbers that were actually NOT in the winning draw
    let winningKilled = 0; // Bad! We killed a winning number
    
    let drawCount = 0;
    
    for (let i = 100; i < allData.length; i++) {
        const hist = allData.slice(0, i);
        const actual = new Set(allData[i]);
        const maxNum = 45;
        
        let killed = new Set();
        
        // RULE 1: Fatigue (Over-consecutive)
        // If a number appeared in last 3 draws, kill it (very rare to hit 4x)
        for (let num = 1; num <= maxNum; num++) {
            if (hist[hist.length-1].includes(num) && 
                hist[hist.length-2].includes(num) && 
                hist[hist.length-3].includes(num)) {
                killed.add(num);
            }
        }
        
        // RULE 2: Markov Dead-End
        // If P(num | ANY number in last draw) is exactly 0 in history
        const tr = {};
        for (let j = 0; j < hist.length - 1; j++) {
            for (const x of hist[j]) {
                if (!tr[x]) tr[x] = new Set();
                for (const y of hist[j+1]) {
                    tr[x].add(y);
                }
            }
        }
        
        const lastDraw = hist[hist.length-1];
        for (let num = 1; num <= maxNum; num++) {
            let possible = false;
            for (const prev of lastDraw) {
                if (tr[prev] && tr[prev].has(num)) {
                    possible = true;
                    break;
                }
            }
            // If none of the numbers in last draw have ever transitioned to `num`
            if (!possible) {
                // killed.add(num); // Wait, this might be too aggressive, history might just be missing it. Let's test.
            }
        }
        
        // RULE 3: Extreme Cold + No Momentum
        // Gap > 20 and appeared less than 2 times in last 50 draws
        const gaps = {};
        for(let j=0; j<hist.length; j++) {
            for(const n of hist[j]) gaps[n] = hist.length - j;
        }
        for (let num = 1; num <= maxNum; num++) {
            const gap = gaps[num] || hist.length;
            if (gap > 20) {
                let count50 = 0;
                for (let j = Math.max(0, hist.length - 50); j < hist.length; j++) {
                    if (hist[j].includes(num)) count50++;
                }
                if (count50 < 2) {
                    killed.add(num);
                }
            }
        }
        
        // RULE 4: Out of Phase (Too many numbers from same decade recently)
        // If a decade (e.g. 1-10) had 4+ numbers in last draw, it usually cools down
        const decadeCount = [0, 0, 0, 0, 0];
        for (const n of lastDraw) decadeCount[Math.floor((n-1)/10)]++;
        for (let d = 0; d < 5; d++) {
            if (decadeCount[d] >= 4) {
                // Kill the whole decade in next draw? Too aggressive. 
                // Let's kill the cold numbers in that decade.
                for (let n = d*10 + 1; n <= Math.min(maxNum, d*10 + 10); n++) {
                    if (!lastDraw.includes(n) && (gaps[n] || 0) > 10) {
                        killed.add(n);
                    }
                }
            }
        }
        
        // Count stats
        totalKilled += killed.size;
        drawCount++;
        
        let killedWin = 0;
        for (const num of killed) {
            if (actual.has(num)) {
                winningKilled++;
                killedWin++;
            } else {
                safeKilled++;
            }
        }
    }
    
    console.log(`Tested on ${drawCount} draws`);
    console.log(`Average numbers eliminated per draw: ${(totalKilled / drawCount).toFixed(1)}`);
    console.log(`Total Eliminations: ${totalKilled}`);
    console.log(`Safe Eliminations (Noise removed): ${safeKilled} (${(safeKilled/totalKilled*100).toFixed(2)}%)`);
    console.log(`FATAL Eliminations (Killed winning num): ${winningKilled} (${(winningKilled/totalKilled*100).toFixed(2)}%)`);
}

async function main() {
    const data = await fetchData();
    runElimination(data);
}
main();
