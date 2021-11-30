/*
|--------------------------------------------------------------------------
| Routes
|--------------------------------------------------------------------------
|
| This file is dedicated for defining HTTP routes. A single file is enough
| for majority of projects, however you can define routes in different
| files and just make sure to import them inside this file. For example
|
| Define routes in following two files
| ├── start/routes/cart.ts
| ├── start/routes/customer.ts
|
| and then import them inside `start/routes.ts` as follows
|
| import './routes/cart'
| import './routes/customer''
|
*/

import Route from '@ioc:Adonis/Core/Route'

Route.get('/', async ({ view }) => {
  return view.render('home')
})

Route.get('/processingSubmission', async ({ view }) => {
  return view.render('wait')
})

Route.get('/document/:filename/:pageNumber', async ({ view, request }) => {
  const filename = request.param('filename');
  const pageNumber = request.param('pageNumber')
  console.log(filename, pageNumber)
  return view.render('document', { filename, pageNumber })
})

Route.post('/submitFile', async ({ request, response }) => {
  response.redirect('/processingSubmission')
})

Route.get('/edition/:editionId', async ({ view }) => {
  return view.render('edition')
})
