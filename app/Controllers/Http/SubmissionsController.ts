import { readFile } from 'fs/promises'
import { existsSync } from 'fs'
import { spawn } from 'child_process'
import glob from 'glob'
import { createHash } from 'crypto'
import { HttpContextContract } from '@ioc:Adonis/Core/HttpContext'
import File from 'App/Models/File'
import Job from 'App/Models/Job'
import Application from '@ioc:Adonis/Core/Application'
import Output from 'App/Models/Output'
import Env from '@ioc:Adonis/Core/Env'

export default class SubmissionsController {
  public async index(ctx: HttpContextContract) {
    return ctx.view.render('home')
  }

  public async submit(ctx: HttpContextContract) {
    const file = ctx.request.file('file', { size: '20mb', extnames: ['pdf'] })
    if (file === null || file.hasErrors) {
      return ctx.response.badRequest('Por favor, envie um arquivo PDF válido com no máximo 20MB')
    }

    const fileContents = await readFile(file.tmpPath!)
    
    const fileHash = createHash('md5').update(fileContents).digest('hex')
    const foundFile = await File.findBy('file_hash', fileHash)
    if (foundFile !== null) {
      return ctx.response.redirect(`/document/${foundFile.id}`)
    }

    const fileSub = await File.create({
      fileName: file.clientName,
      fileHash: fileHash,
      fileContent: fileContents,
      submitterIp: ctx.request.ip(),
    })

    // TODO use env var
    const inputDir = Env.get('OCR_INPUT_PATH')
    const outputDir = Env.get('OCR_OUTPUT_PATH')
    await file.move(`${inputDir}/${fileHash}`, { name: `${fileHash}.pdf` })
    // await file.move(Application.tmpPath('uploads'))
    const job = await Job.create({
      finished: false,
      pageCount: 0,
      outputPath: `${outputDir}/${fileHash}`, // TODO use env var
    })
    await job.related('file').associate(fileSub)

    ctx.logger.info(file.filePath!)
    // TODO use env var
    const ocrDir = Env.get('OCR_PATH')
    const sp = spawn('python3', [`${ocrDir}/main.py`, '-p', '--pdf', file.filePath!, '-o', job.outputPath], { cwd: ocrDir })
    sp.stdout.on('data', (data) => {
      ctx.logger.info(`ocr> ${data}`)
    })
    sp.stderr.on('data', (data) => {
      ctx.logger.error(`ocr> ${data}`)
    })

    return ctx.response.redirect(`/processingSubmission?jobId=${job.id}`)
  }
  
  public async checkCompletion(ctx: HttpContextContract) {
    const jobId = ctx.request.input('jobId')
    if (jobId === null) {
      return ctx.response.badRequest('Por favor, informe qual job quer verificar o status.')
    }
    
    const job = await Job.find(jobId)
    if (job === null) {
      return ctx.response.badRequest(`Não foi possível encontrar um job com o id ${jobId}`)
    }
    await job.load('file')
    
    // always run
    ctx.logger.info(`job.pageCount: ${job.pageCount}`)
    if (job.pageCount === null || job.pageCount === 0) {
      glob(`${job.outputPath}/page*`, (err, pages) => {
        if (err) ctx.logger.error(err.message)

        ctx.logger.info(`pages: ${pages}`)
        
        job.pageCount = pages.length
        job.save()
      })
    }
    
    let exists = job.pageCount > 0
    for (let pg = 1; pg <= job.pageCount; pg++) {
      exists = exists && existsSync(`${job.outputPath}/page0001-${pg}/proc.txt`)
      if (exists) {
        const textContent = await readFile(`${job.outputPath}/page0001-${pg}/proc.txt`, { encoding: 'utf-8' })
        // TODO use env var
        const ocrDir = Env.get('OCR_PATH')
        const imgContent = await readFile(`${ocrDir}/input/processed/${job.file.fileHash}/page0001-${pg}.png`)
        const output = await Output.create({
          pageNumber: pg,
          text: textContent,
          pageImage: imgContent,
        })
        output.related('file').associate(job.file)
      }
    }

    if (exists) {
      job.finished = exists
      job.save()
    }
    
    if (job.finished) {
      return ctx.response.redirect(`/document/${job.file.id}`)
    }
    
    return ctx.view.render('wait')
  }
}
