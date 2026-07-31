/**
 * Memetakan durasi tidur (jam) ke skala kualitas tidur (1 - 10).
 * 
 * Logic Pemetaan:
 * - Titik ideal: 7 - 9 jam -> Nilai 10
 * - Kurang tidur: Menurun 1.5 poin per jam di bawah 7 jam
 * - Kelebihan tidur: Menurun 1.0 poin per jam di atas 9 jam
 * 
 * @param {number} sleepHours - Jumlah jam tidur (bisa desimal, misal: 6.5)
 * @returns {number} Nilai kualitas tidur (Integer 1 - 10)
 */
function mapSleepHoursToQuality(sleepHours) {
    const hours = parseFloat(sleepHours);
    if (isNaN(hours) || hours <= 0) {
        return 1;
    }

    let quality;

    if (hours >= 7 && hours <= 9) {  // Rentang ideal (7 - 9 jam)
        quality = 10;
    } else if (hours < 7) { // Kurang dari 7 jam
        quality = 10 - (7 - hours) * 1.5;
    } else { // Lebih dari 9 jam
        quality = 10 - (hours - 9) * 1.0;
    }

    let roundedQuality = Math.round(quality);

    if (roundedQuality > 10) roundedQuality = 10;
    if (roundedQuality < 1) roundedQuality = 1;

    return roundedQuality;
}

module.exports = mapSleepHoursToQuality;