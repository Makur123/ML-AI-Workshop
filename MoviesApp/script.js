const APILINK = 'https://api.themoviedb.org/3/discover/movie?sort_by=popularity.desc&api_key=23e0e15bffc9eae7fd9b01e076b2b9f7&page=1';
const IMG_PATH = 'https://image.tmdb.org/t/p/w1280';
const SEARCHAPI = "https://api.themoviedb.org/3/search/movie?&api_key=23e0e15bffc9eae7fd9b01e076b2b9f7&query=";

const main = document.getElementById("section");
const form = document.getElementById("form");
const search = document.getElementById("query");

function returnMovies(url) {
    fetch(url)
        .then(res => res.json())
        .then(function(data) {
            console.log(data.results);
            main.innerHTML = ''; // Clear previous results
            data.results.forEach(element => {
                const div_card = document.createElement('div');
                div_card.setAttribute('class', 'card');

                const div_row = document.createElement('div');
                div_row.setAttribute('class', 'row');

                const div_column = document.createElement('div');
                div_column.setAttribute('class', 'column');

                const image = document.createElement('img');
                image.setAttribute('class', 'thumbnail');
                image.setAttribute('id', 'image');

                const title = document.createElement('h3');
                title.setAttribute('id', 'title');
                const center = document.createElement('center');

                title.innerHTML = `${element.title}`;
                image.src = IMG_PATH + element.poster_path;

                center.appendChild(image);
                div_card.appendChild(center);
                div_card.appendChild(title);

                div_column.appendChild(div_card);
                div_row.appendChild(div_column);
                main.appendChild(div_row); // Append to the main section
            });
        })
        .catch(error => console.error('Error:', error));
}

// Add event listener to the form
form.addEventListener('submit', (e) => {
    e.preventDefault();
    const searchTerm = search.value;
    if (searchTerm) {
        returnMovies(SEARCHAPI + searchTerm);
        search.value = ''; // Clear the search input
    }
});

// Initial call to display popular movies
returnMovies(APILINK);