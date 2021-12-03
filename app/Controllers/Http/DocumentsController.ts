import { HttpContextContract } from '@ioc:Adonis/Core/HttpContext'
import Output from 'App/Models/Output'
import File from 'App/Models/File';
import Job from 'App/Models/Job';
import { readFile } from 'fs/promises';

export default class DocumentsController {
  public async viewPage(ctx: HttpContextContract) {
    const fileId = ctx.request.param('fileId');
    if (fileId === null) {
      return ctx.response.notFound()
    }
    const fileSub = await File.find(fileId)
    if (fileSub === null) {
      return ctx.response.notFound()
    }
    const pageNumber = Number.parseInt(ctx.request.param('pageNumber', '1'))

    const job = await Job.findBy('fileId', fileId)
    if (job === null) {
      return ctx.response.notFound()
    }

    const out = await Output.query().where('file_id', fileId).andWhere('page_number', pageNumber).first()
    if (out === null) {
      return ctx.response.notFound()
    }

    const pageCount = job.pageCount

    const previousPage = pageNumber > 0 ? pageNumber - 1 : null
    const nextPage = pageNumber < pageCount ? pageNumber + 1 : null

    return ctx.view.render('document', { fileName: fileSub.fileName, pageNumber, previousPage, nextPage, image: `/image/${fileId}/${pageNumber}`, text: `/text/${fileId}/${pageNumber}` })
  }

  public async getText(ctx: HttpContextContract) {
    const fileId = ctx.request.param('fileId');
    if (fileId === null) {
      return ctx.response.notFound()
    }
    const fileSub = await File.find(fileId)
    if (fileSub === null) {
      return ctx.response.notFound()
    }
    const pageNumber = Number.parseInt(ctx.request.param('pageNumber', '1'))
    
    const out = await Output.query().where('file_id', fileId).andWhere('page_number', pageNumber).first()
    if (out === null) {
      return ctx.response.notFound()
    }
    
    ctx.response.header('content-disposition', `inline; filename="${fileSub.fileHash}.txt"`)
    ctx.response.type('.txt')
    return out.text
  }

  public async getImage(ctx: HttpContextContract) {
    const fileId = ctx.request.param('fileId');
    if (fileId === null) {
      return ctx.response.notFound()
    }
    const fileSub = await File.find(fileId)
    if (fileSub === null) {
      return ctx.response.notFound()
    }
    const pageNumber = Number.parseInt(ctx.request.param('pageNumber', '1'))
    
    const out = await Output.query().where('file_id', fileId).andWhere('page_number', pageNumber).first()
    if (out === null) {
      return ctx.response.notFound()
    }

    ctx.response.header('content-disposition', `inline; filename="${fileSub.fileHash}.png"`)
    ctx.response.type('.png')
    return out.pageImage
  }
}
