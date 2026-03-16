document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[2],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[3],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[4],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[5],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[6],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[7],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[8],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[9],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[10],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        

        function n3vSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 6
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/coffee_martini/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/cook_spinach/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/cut_roasted_beef/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/flame_salmon/';
                        break;
                    case 4:
                        image.src = 'static/resources/comp/flame_steak/';
                        break;
                    case 5:
                        image.src = 'static/resources/comp/sear_steak/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'MoE.png';
                        break;
                    case 1:
                        image.src = image.src + '4DGaussians.png';
                        break;
                    case 2:
                        image.src = image.src + '4DGS.png';
                        break;
                    case 3:
                        image.src = image.src + 'STG.png';
                        break;
                    case 4:
                        image.src = image.src + 'Ex4DGS.png';
                        break;
                    case 5:
                        image.src = image.src + 'E-D3DGS.png';
                        break;

                }
            }
            
            

            let scene_list = document.getElementById("object-scale-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function coffeeSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/n3v/4DGaussians/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/n3v/STG/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/n3v/Ex4DGS/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/n3v/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'coffee_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'coffee.png';
                        break;
                }
            }

            let scene_list = document.getElementById("n3v-coffee").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function theaterSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[1]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/technicolor/STG/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/technicolor/Ex4DGS/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/technicolor/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'theater_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'theater.png';
                        break;
                }
            }

            let scene_list = document.getElementById("technicolor-theater").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function spinachSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[9]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/n3v/4DGaussians/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/n3v/4DGS/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/n3v/STG/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/n3v/Ex4DGS/';
                        break;
                    case 4:
                        image.src = 'static/resources/comp/n3v/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'spinach_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'spinach.png';
                        break;
                }
            }

            let scene_list = document.getElementById("n3v-spinach").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function fabienSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[2]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/technicolor/Ex4DGS/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/technicolor/STG/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/technicolor/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'fabien_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'fabien.png';
                        break;
                }
            }

            let scene_list = document.getElementById("technicolor-fabien").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function painterSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[3]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/technicolor/Ex4DGS/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/technicolor/STG/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/technicolor/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'painter_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'painter.png';
                        break;
                }
            }

            let scene_list = document.getElementById("technicolor-painter").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function beefSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[2]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/n3v/4DGaussians/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/n3v/4DGS/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/n3v/STG/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/n3v/Ex4DGS/';
                        break;
                    case 4:
                        image.src = 'static/resources/comp/n3v/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'beef_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'beef.png';
                        break;
                }
            }

            let scene_list = document.getElementById("n3v-beef").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function salmonSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[3]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/n3v/4DGaussians/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/n3v/4DGS/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/n3v/STG/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/n3v/Ex4DGS/';
                        break;
                    case 4:
                        image.src = 'static/resources/comp/n3v/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'salmon_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'salmon.png';
                        break;
                }
            }

            let scene_list = document.getElementById("n3v-salmon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function flamesteakSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[4]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/n3v/4DGaussians/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/n3v/4DGS/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/n3v/STG/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/n3v/Ex4DGS/';
                        break;
                    case 4:
                        image.src = 'static/resources/comp/n3v/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'flamesteak_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'flamesteak.png';
                        break;
                }
            }

            let scene_list = document.getElementById("n3v-flamesteak").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function searsteakSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[5]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/n3v/4DGaussians/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/n3v/4DGS/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/n3v/STG/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/n3v/Ex4DGS/';
                        break;
                    case 4:
                        image.src = 'static/resources/comp/n3v/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'searsteak_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'searsteak.png';
                        break;
                }
            }

            let scene_list = document.getElementById("n3v-searsteak").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function birthdaySceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[7]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/technicolor/Ex4DGS/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/technicolor/STG/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/technicolor/E-D3DGS/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'birthday_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'birthday.png';
                        break;
                }
            }

            let scene_list = document.getElementById("technicolor-birthday").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        

        

        function trainSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[10]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/technicolor/Ex4DGS/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/technicolor/STG/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/technicolor/E-D3DGS/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/n3v/GT/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'train_MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'train.png';
                        break;
                }
            }

            let scene_list = document.getElementById("technicolor-train").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function TechnicolorSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 4
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'static/resources/comp/Birthday/';
                        break;
                    case 1:
                        image.src = 'static/resources/comp/Fabien/';
                        break;
                    case 2:
                        image.src = 'static/resources/comp/Painter/';
                        break;
                    case 3:
                        image.src = 'static/resources/comp/Theater/';
                        break;
                    case 4:
                        image.src = 'static/resources/comp/Train/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'MoE.png';
                        break;
                    case 1:
                        image.src = image.src + 'STG.png';
                        break;
                    case 2:
                        image.src = image.src + 'Ex4DGS.png';
                        break;
                    case 3:
                        image.src = image.src + 'E-D3DGS.png';
                        break;
                }
            }
            

            let scene_list = document.getElementById("object-scale-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }
