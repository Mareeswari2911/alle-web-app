const sharp = require("sharp");


async function getPixels(buffer) {
  return await sharp(buffer)
    .raw()
    .toBuffer({ resolveWithObject: true });
}
function calculatePSNR(mse) {
  if (mse === 0) return 100;
  const maxPixel = 255;
  return 20 * Math.log10(maxPixel / Math.sqrt(mse));
}
function calculateSSIM(img1, img2) {
  let mean1 = 0, mean2 = 0;

  for (let i = 0; i < img1.length; i++) {
    mean1 += img1[i];
    mean2 += img2[i];
  }

  mean1 /= img1.length;
  mean2 /= img2.length;

  let var1 = 0, var2 = 0, cov = 0;

  for (let i = 0; i < img1.length; i++) {
    var1 += Math.pow(img1[i] - mean1, 2);
    var2 += Math.pow(img2[i] - mean2, 2);
    cov += (img1[i] - mean1) * (img2[i] - mean2);
  }

  var1 /= img1.length;
  var2 /= img2.length;
  cov /= img1.length;

  const c1 = 0.01 * 255 * 0.01 * 255;
  const c2 = 0.03 * 255 * 0.03 * 255;

  const ssim =
    ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) /
    ((mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2));

  return ssim;
}

module.exports = {
  calculatePSNR,
  calculateSSIM,
  getPixels
};