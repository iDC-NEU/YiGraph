!(function (MainApp) {
    "use strict";
    
    
    /*  =======================================================
      Custom Menu (sidebar/header)
    ========================================================== */
    let menu={
      classes: {
        main: 'menu-side',
        item:'menu-item',
        link:'menu-link',
        toggle: 'menu-toggle',
        sub: 'sub-menu',
        subparent: 'has-sub',
        active: 'is-active',
        current: 'current-page'
      },
    };
    let header_menu={
      classes: {
        main: 'menu-head',
        item:'menu-item',
        link:'menu-link',
        toggle: 'has-toggle',
        sub: 'sub-menu',
        subparent: 'has-sub',
        active: 'active',
        current: 'current-page'
      },
    };
    
    
    MainApp.Menu = {
      load: function(elm,subparent){
        let parent = elm.parentElement;
        if(!parent.classList.contains(subparent)){
          parent.classList.add(subparent);
        }
      },
      toggle: function(elm,active){
        let parent = elm.parentElement;
        let nextelm = elm.nextElementSibling;
        let speed = nextelm.children.length > 5 ? 400 + nextelm.children.length * 10 : 400;
        if(!parent.classList.contains(active)){
          parent.classList.add(active);
          MainApp.SlideDown(nextelm,speed);
        }else{
          parent.classList.remove(active);
          MainApp.SlideUp(nextelm,speed);
        }
      },
      closeSiblings: function(elm,active,subparent,submenu){
        let parent = elm.parentElement;
        let siblings = parent.parentElement.children;
        Array.from(siblings).forEach(item => {
          if(item !== parent){
            item.classList.remove(active);
            if(item.classList.contains(subparent)){
              let subitem = item.querySelectorAll(`.${submenu}`);
              subitem.forEach(child => {
                child.parentElement.classList.remove(active);
                MainApp.SlideUp(child,400);
              })
            }
          }
        });
      }
    }

    //init Menu
    MainApp.Menu.sidebar = function (){
      const elm = document.querySelectorAll(`.${menu.classes.toggle}`);
      let active = menu.classes.active;
      let subparent = menu.classes.subparent;
      let submenu = menu.classes.sub;
      elm?.forEach(item => {
        MainApp.Menu.load(item,subparent);
        item.addEventListener("click", function(e){
          e.preventDefault();
          MainApp.Menu.toggle(item,active);
          MainApp.Menu.closeSiblings(item,active,subparent,submenu);
        });
      })
    };
    MainApp.Menu.header = function (){
      const elm = document.querySelectorAll(`.${header_menu.classes.toggle}`);
      let active = header_menu.classes.active;
      let subparent = header_menu.classes.subparent;
      let submenu = header_menu.classes.sub;
      elm?.forEach(item => {
        MainApp.Menu.load(item,subparent);
        item.addEventListener("click", function(e){
          e.preventDefault();
          if(MainApp.Win.width < MainApp.Break.xl){
            MainApp.Menu.toggle(item,active);
            MainApp.Menu.closeSiblings(item,active,subparent,submenu);
          }
        });
      })
    };


    /*  =======================================================
      Custom Sidebar
    ========================================================== */
    MainApp.Sidebar ={
      toggle: function(){
        let toggle = document.querySelectorAll('.sidebar-toggle');
        let target = document.querySelector('.main-sidebar');
        toggle.forEach(item => {
          item.addEventListener("click", function(e){
            e.preventDefault();
            toggle.forEach(item => {
              item.classList.toggle('active');
            })
            target?.classList.toggle('sidebar-visible');
            document.body.classList.toggle('overflow-hidden')
          });
        })
      },

      page_resize: function(){
        let toggle = document.querySelectorAll('.sidebar-toggle');
        let target = document.querySelector('.main-sidebar');
        if(MainApp.Win.width > MainApp.Break.xl2) {
          toggle.forEach(item => {
            item.classList.remove('active');
          })
          target?.classList.remove('sidebar-visible');
          document.body.classList.remove('overflow-hidden');
        }
      }
    }

    MainApp.Sidebar.init = function (){
      MainApp.Sidebar.toggle();
      window.addEventListener('resize', function(){
        MainApp.Sidebar.page_resize();
      });
    }

    /*  =======================================================
      Custom Header
    ========================================================== */
    MainApp.Header ={
      toggle: function(){
        let toggle = document.querySelectorAll('.header-toggle');
        let target = document.querySelector('.main-header');
        toggle.forEach(item => {
          item.addEventListener("click", function(e){
            e.preventDefault();
            toggle.forEach(item => {
              item.classList.toggle('active');
            })
            target?.classList.toggle('header-visible');
            document.body.classList.toggle('overflow-hidden')
          });
        })
      },

      page_resize: function(){
        let toggle = document.querySelectorAll('.header-toggle');
        let target = document.querySelector('.main-header');
        if(MainApp.Win.width > MainApp.Break.xl) {
          toggle.forEach(item => {
            item.classList.remove('active');
          })
          target?.classList.remove('header-visible');
          document.body.classList.remove('overflow-hidden');
        }
      }
    }

    MainApp.Header.init = function (){
      MainApp.Header.toggle();
      window.addEventListener('resize', function(){
        MainApp.Header.page_resize();
      });
    }
    
    /*  =======================================================
      Add some class to current link
    ========================================================== */
    MainApp.CurrentLink = function(selector, parent, submenu, base, active){
      let elm = document.querySelectorAll(selector);
      let currentURL = document.location.href,
      removeHash = currentURL.substring(0, (currentURL.indexOf("#") == -1) ? currentURL.length : currentURL.indexOf("#")),
      removeQuery = removeHash.substring(0, (removeHash.indexOf("?") == -1) ? removeHash.length : removeHash.indexOf("?")),
      fileName = removeQuery;
      
      elm.forEach(function(item){
        var selfLink = item.getAttribute('href').split('../').slice(-1);
        if (fileName.match(selfLink)) {
          let parents = MainApp.getParents(item,`.${base}`, parent);
          parents.forEach(parentElemets =>{
            parentElemets.classList.add(...active);
            let subItem = parentElemets.querySelector(`.${submenu}`);
            subItem !== null && (subItem.style.display = "block")
            parentElemets.scrollIntoView({ behavior: "smooth", block: "start",inline: "nearest"});
          })
        } else {
          item.parentElement.classList.remove(...active);
        }
      })
    }
    MainApp.CurrentLinkApp = function(selector, parent, submenu, base, active){
      let elm = document.querySelectorAll(selector);
      let currentURL = document.location.href,
      removeHash = currentURL.substring(0, (currentURL.indexOf("#") == -1) ? currentURL.length : currentURL.indexOf("#")),
      fileName = removeHash;
      elm.forEach(function(item){
        var selfLink = item.getAttribute('href').replace('../', '').replace('./', '');
        if (fileName.includes(selfLink)) {
          let parents = MainApp.getParents(item,`.${base}`, parent);
          parents.forEach(parentElemets =>{
            parentElemets.classList.add(...active);
            let subItem = parentElemets.querySelector(`.${submenu}`);
            subItem !== null && (subItem.style.display = "block")
            parentElemets.scrollIntoView({ behavior: "smooth", block: "start",inline: "nearest"});
          })
        } else {
          item.parentElement.classList.remove(...active);
        }
      })
    }

    /*  ================================================================
      Custom select js (Choices)
    ==================================================================== */
    MainApp.Select = function(selector,options){
      let elm = document.querySelectorAll(selector);
      if( elm.length > 0 ){
        elm.forEach(item => {
          let search = item.dataset.search ? JSON.parse(item.dataset.search) : false;
          let sort = item.dataset.sort ? JSON.parse(item.dataset.sort) : false;
          let cross = item.dataset.cross ? JSON.parse(item.dataset.cross) : true;
          let placeholderValue = item.dataset.placeholder ? item.dataset.placeholder : null;
          const choices = new Choices(item, {
            silent: true,
            allowHTML: false,
            searchEnabled: search,
            placeholder: true,
            placeholderValue: placeholderValue,
            searchPlaceholderValue: '',
            shouldSort: sort,
            removeItemButton: cross,
            itemSelectText: '',
            noResultsText: 'No results',
          });
        })
      }
    }

    /*  ================================================================
      Drag and drop input/upload field (dropzone)
    ==================================================================== */
    MainApp.Dropzone = function(selector){
      let elm = document.querySelectorAll(selector);
      if( elm != 'undefined' && elm != null ){
        elm.forEach(item => {
          let itemId = item.id;
          let maxFiles = item.dataset.maxFiles ? parseInt(item.dataset.maxFiles) : null;
          let maxFilesize = item.dataset.maxFilesize ? parseInt(item.dataset.maxFilesize) : 256;
          let acceptedFiles = item.dataset.acceptedFiles ? item.dataset.acceptedFiles : null;

          //add styling Class 
          item.classList.add('dropzone');
          
          let myDropzone = new Dropzone(`#${itemId}`,{
            url: "image",
            maxFilesize: maxFilesize,
            maxFiles: maxFiles,
            acceptedFiles: acceptedFiles
          });
        })
      }
    }

    /*  ================================================================
      Slider (swiper)
    ==================================================================== */
    MainApp.Slider = function(selector){
      let elm = document.querySelectorAll(selector);
      if( elm != 'undefined' && elm != null ){
        elm.forEach(item => {
          let _breakpoints = item.dataset.breakpoints ? JSON.parse(item.dataset.breakpoints) : null;
          let _autoplay = item.dataset.autoplay ? JSON.parse(item.dataset.autoplay) : false;
          let _loop = item.dataset.loop ? JSON.parse(item.dataset.loop) : false;
          let _centeredSlides = item.dataset.centeredslides ? JSON.parse(item.dataset.centeredslides) : false;
          let _speed = item.dataset.speed ? parseInt(item.dataset.speed) : 1000;
          let _spaceBetween = item.dataset.spaceBetween ? parseInt(item.dataset.spaceBetween) : 0;
          let _effect = item.dataset.effect ? item.dataset.effect : '';
          let _parent = item.dataset.parent ? item.dataset.parent : false;
          var swiper = new Swiper(item, {
            // Optional parameters
            centeredSlides: _centeredSlides,
            loop: _loop,
            speed: _speed,
            autoplay:_autoplay,
            effect: _effect,
            spaceBetween: _spaceBetween,
            // If we need pagination
            pagination: {
              el: ".swiper-pagination",
              type: 'bullets',
              clickable: true,
            },
            // Navigation arrows
            navigation: {
              nextEl: '.swiper-button-next',
              prevEl: '.swiper-button-prev',
              clickable: true,
            },
            breakpoints: _breakpoints,
          });

          if(_parent){
            let item = document.querySelector(_parent);
            let _breakpoints = item.dataset.breakpoints ? JSON.parse(item.dataset.breakpoints) : null;
            let _autoplay = item.dataset.autoplay ? JSON.parse(item.dataset.autoplay) : false;
            let _loop = item.dataset.loop ? JSON.parse(item.dataset.loop) : false;
            let _centeredSlides = item.dataset.centeredslides ? JSON.parse(item.dataset.centeredslides) : false;
            let _speed = item.dataset.speed ? parseInt(item.dataset.speed) : 1000;
            let _spaceBetween = item.dataset.spaceBetween ? parseInt(item.dataset.spaceBetween) : 0;
            let _effect = item.dataset.effect ? item.dataset.effect : '';
            var swiper_parent = new Swiper(item, {
              // Optional parameters
              centeredSlides: _centeredSlides,
              loop: _loop,
              speed: _speed,
              autoplay:_autoplay,
              effect: _effect,
              spaceBetween: _spaceBetween,
              // If we need pagination
              pagination: {
                el: ".swiper-pagination",
                type: 'bullets',
                clickable: true,
              },
              // Navigation arrows
              navigation: {
                nextEl: '.swiper-button-next',
                prevEl: '.swiper-button-prev',
                clickable: true,
              },
              breakpoints: _breakpoints,
              thumbs: {
                swiper: swiper,
              },
            });
          }

          
        });
      }
    }
    
    /*  =======================================================
      Editor
    ========================================================== */
    MainApp.Quill = function (selector, opt) {
      let elm = document.querySelectorAll(selector);
        if( elm != 'undefined' && elm != null ){
          elm.forEach(item => {
            let toolbarConfig = [
              [{ 'header': [ 3, 4, 5, 6, false] }],
              ['bold', 'italic', 'underline'],
              [ 'blockquote' ,{ 'list': 'bullet' }],
              [{ 'align': [] }],
              ['clean']
          ]
            let quill = new Quill(item, {
                modules: {
                    toolbar: toolbarConfig
                },
                theme: 'snow'
            });
          })
        }
    }

    /*  =======================================================
      Lightbox
    ========================================================== */
    MainApp.Lightbox = function () {
      let elm = document.querySelectorAll('.js-lightbox-toggle');
      if( elm != 'undefined' && elm != null ){
        elm.forEach(item => {
          const lightbox = new PhotoSwipeLightbox({
            gallery: '.js-lightbox-gallery',
            children: '.js-lightbox-toggle',
            maxWidthToAnimate: 10,
            pswpModule: PhotoSwipe 
          });
          lightbox.init();
        })
      }
    }

    /*  =======================================================
      SyntaxHighlight
    ========================================================== */
    MainApp.SyntaxHighlight = function () {
      let elm = document.querySelectorAll('pre code');
      if( elm != 'undefined' && elm != null ){
        elm.forEach(item => {
          hljs.highlightElement(item);
        })
      }
    }
    /*  =======================================================
      Clipboard
    ========================================================== */
    MainApp.Clipboard = function() {
      let clipboardTrigger = document.querySelectorAll('.js-copy');
      let options = {
        tooltip:{
          init: 'Copy',
          success : 'Copied',
        }
      }
      clipboardTrigger.forEach(item => {
        //init clipboard
        let clipboard = new ClipboardJS(item);
        //set markup
        let initMarkup = `${options.tooltip.init}`;
        let successMarkup = `${options.tooltip.success}`;
        item.innerHTML = initMarkup;
        //on-sucess
        clipboard.on("success", function(e){
          let target = e.trigger;
          target.classList.add('success')
          target.innerHTML = successMarkup;
          setTimeout(function(){
            target.innerHTML = initMarkup;
            target.classList.remove('success')
          }, 1000)
        });
      });
    }

    /*  ================================================================
      Typewriter (typewriter)
    ==================================================================== */
    MainApp.Typewriter = function(selector){
      let elm = document.querySelectorAll(selector);
      if( elm != 'undefined' && elm != null ){
        elm.forEach(item => {
          let _strings = item.dataset.strings ? JSON.parse(item.dataset.strings) : null;
          let _autostart = item.dataset.autostart ? JSON.parse(item.dataset.autostart) : true;
          let _loop = item.dataset.loop ? JSON.parse(item.dataset.loop) : true;
          new Typewriter(item, {
            strings: _strings,
            autoStart: _autostart,
            loop: _loop
          });
        });
      }
    }

    /*  ================================================================
      Typewriter (typewriter)
    ==================================================================== */
    MainApp.Masonry = function(selector){
      let elm = document.querySelectorAll(selector);
      if( elm != 'undefined' && elm != null ){
        elm.forEach(item => {
          var msnry = new Masonry( item, {
            itemSelector: '.masonry-item'
          });
        });
      }
    }

    /*  ================================================================
      AutoChangeInput
    ==================================================================== */
    MainApp.AutoChangeInput = function (selector) {
      let elem = document.querySelectorAll(selector);
      if (elem) {
        elem.forEach(item => {
          item.onkeyup = function(e) {
            var target = e.srcElement;
            var maxLength = parseInt(target.attributes["maxlength"].value, 10);
            var myLength = target.value.length;
            if (myLength >= maxLength) {
                var next = target;
                while (next = next.nextElementSibling) {
                    if (next == null)
                        break;
                    if (next.tagName.toLowerCase() == "input") {
                        next.focus();
                        break;
                    }
                }
            }
            if(e.key === "Backspace"){
              var previous = target;
              while (previous = previous.previousElementSibling) {
                if (previous == null)
                    break;
                if (previous.tagName.toLowerCase() == "input") {
                    previous.focus();
                    break;
                }
              }
            }
          }
        })
      }
    }
    
    /*  =======================================================
      Mode Switch
    ========================================================== */
    const defaultTheme = {
      mode: "light",
      direction : 'ltr'
    }
    let theme = [];
    let date = new Date().toLocaleDateString();
    MainApp.Settings = function () {
      if(localStorage.getItem("scribbler-html")){
          let localData = JSON.parse(localStorage.getItem("scribbler-html"));
          if (localData.date !== date) {
              theme = defaultTheme;
          } else {
              theme = theme;
          }
      }else{
          localStorage.setItem("scribbler-html", JSON.stringify({ ...defaultTheme, date: date }));
      }

      function onPageLoad(){
          let themeData = JSON.parse(localStorage.getItem("scribbler-html"));
          if(themeData.mode === 'dark'){
              document.documentElement.classList.add('dark');
          }else{
              document.documentElement.classList.remove('dark');
          }
          if(themeData.direction === 'rtl'){
              document.body.setAttribute('dir', 'rtl');
          }else{
              document.body.setAttribute('dir', 'ltr');
          }
      }
      onPageLoad();


      const themeToggle = document.querySelectorAll('.theme-toggle');
      const directionToggle = document.querySelectorAll('.direction-toggle');

      // mode toggle
      themeToggle.forEach((item)=> {
          item.addEventListener('click', function(e){
              e.preventDefault();
              let themeData = JSON.parse(localStorage.getItem("scribbler-html"));
              if(themeData.mode !== 'dark'){
                  localStorage.setItem("scribbler-html", JSON.stringify({ ...themeData, mode: 'dark', date: date }))
                  document.documentElement.classList.add('dark');
              }else{
                  localStorage.setItem("scribbler-html", JSON.stringify({ ...themeData, mode: 'light', date: date }))
                  document.documentElement.classList.remove('dark');
              }
          })
      })
      // direction toggle
      directionToggle.forEach((item)=> {
          item.addEventListener('click', function(e){
              e.preventDefault();
              let themeData = JSON.parse(localStorage.getItem("scribbler-html"));
              if(themeData.direction !== 'rtl'){
                  localStorage.setItem("scribbler-html", JSON.stringify({ ...themeData, direction: 'rtl', date: date }))
                  document.body.setAttribute('dir', 'rtl');
              }else{
                  localStorage.setItem("scribbler-html", JSON.stringify({ ...themeData, direction: 'ltr', date: date }))
                  document.body.setAttribute('dir', 'ltr');
              }
          })
      })
    }

    
    /*  =======================================================
      Custom Scripts init 
    ========================================================== */
    MainApp.Custom.init = function () {
      MainApp.Menu.sidebar();
      MainApp.Menu.header();
      MainApp.Sidebar.init();
      MainApp.Header.init();
      MainApp.Select('.js-select');
      MainApp.Dropzone('.js-upload');
      MainApp.Slider('.js-slider');
      MainApp.AutoChangeInput('.js-auto-input-change');
      MainApp.Quill('.js-quill');
      MainApp.Typewriter('.js-typewriter');
      MainApp.Masonry('.js-masonry');
      MainApp.Lightbox();
      MainApp.SyntaxHighlight();
      MainApp.Clipboard();
      MainApp.Settings();
      MainApp.CurrentLink(`.${menu.classes.main} .${menu.classes.link}`, menu.classes.item, menu.classes.sub, menu.classes.main, [menu.classes.active, menu.classes.current]);
      MainApp.CurrentLinkApp(`.${header_menu.classes.main} .${header_menu.classes.link}`, header_menu.classes.item, header_menu.classes.sub, header_menu.classes.main, [header_menu.classes.active, header_menu.classes.current]);
    }
    
    // Initial by default
    /////////////////////////////
    MainApp.init = function () {
      MainApp.winLoad(MainApp.Custom.init);
    }
    
    MainApp.init();
    
    return MainApp;
    })(MainApp);
    