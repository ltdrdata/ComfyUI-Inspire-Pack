import { ComfyApp, app } from "../../scripts/app.js";

function load_image(str) {
	let base64String = canvas.toDataURL('image/png');
	let img = new Image();
	img.src = base64String;
}

app.registerExtension({
	name: "Comfy.Inspire.img",

	nodeCreated(node, app) {
		if(node.comfyClass == "LoadImage //Inspire") {
			console.log(node);
			let w = node.widgets.find(obj => obj.name === 'image_data');

			Object.defineProperty(node, 'imgs', {
				set(v) {
					this._img = v;

					var canvas = document.createElement('canvas');
					canvas.width = v[0].width;
					canvas.height = v[0].height;

					var context = canvas.getContext('2d');
					context.drawImage(v[0], 0, 0, v[0].width, v[0].height);

					var base64Image = canvas.toDataURL('image/png');
					w.value = base64Image;
				},
				get() {
					if(this._img == undefined && w.value != '') {
						this._img = [new Image()];
						this._img[0].src = w.value;
					}

					return this._img;
				}
			});
		}
    }
})