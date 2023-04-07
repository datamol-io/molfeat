var gtagId = "G-23R9EJVY37";

var script = document.createElement("script");
script.src = "https://www.googletagmanager.com/gtag/js?id=" + gtagId;
document.head.appendChild(script);

window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', gtagId);
