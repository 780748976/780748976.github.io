hexo.on('generateAfter', function() {
  hexo.log.info('Creating .nojekyll in public directory');
  var fs = require('fs');
  var path = require('path');
  var publicDir = hexo.public_dir;
  if (!fs.existsSync(publicDir)) {
    fs.mkdirSync(publicDir, { recursive: true });
  }
  fs.writeFileSync(path.join(publicDir, '.nojekyll'), '');
});
