import { readFile } from 'fs/promises'
import { existsSync } from 'fs'
import { spawn } from 'child_process'
import glob from 'glob'
import { createHash } from 'crypto'
import { HttpContextContract } from '@ioc:Adonis/Core/HttpContext'
import File from 'App/Models/File'
import Job from 'App/Models/Job'
import Output from 'App/Models/Output'
import Env from '@ioc:Adonis/Core/Env'
import { promisify } from 'util'

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

    const inputDir = Env.get('OCR_INPUT_PATH')
    const outputDir = Env.get('OCR_OUTPUT_PATH')
    await file.move(`${inputDir}/${fileHash}`, { name: `${fileHash}.pdf` })
    
    const job = await Job.create({
      finished: false,
      failed: false,
      pageCount: 0,
      outputPath: `${outputDir}/${fileHash}`,
    })
    await job.related('file').associate(fileSub)

    const ocrDir = Env.get('OCR_PATH')
    const sp = spawn('python3', [`${ocrDir}/main.py`, '-p', '--pdf', file.filePath!, '-o', job.outputPath], { cwd: ocrDir })
    sp.stdout.on('data', (data) => {
      ctx.logger.info(`[OCR] ${data}`)
    })
    const onError = (err: any) => {
      ctx.logger.error(`[OCR] (ERROR) ${err}`)
      job.failed = true
      job.finished = false
      job.save()
    }
    sp.on('error', onError)

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
    
    const _glob = promisify(glob)
    const pages = await _glob(`${job.outputPath}/page*`)
    job.pageCount = pages.length
    job.save()
    
    let exists = job.pageCount > 0
    for (let pg = 1; pg <= job.pageCount; pg++) {
      exists = exists && existsSync(`${job.outputPath}/page0001-${pg}/proc.txt`)
      if (exists) {
        const textContent = await readFile(`${job.outputPath}/page0001-${pg}/proc.txt`, { encoding: 'utf-8' })
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
      job.failed = false
      job.finished = exists
      job.save()
    }
    
    if (job.finished) {
      return ctx.response.redirect(`/document/${job.file.id}`)
    } else if (job.failed) {
      ctx.logger.info(`job failed? ${job.failed}`)
      job.file.delete()
      return ctx.response.internalServerError(`Job ${jobId} falhou, tente novamente mais tarde ou contate o administrador.`)
    }
    
    return ctx.view.render('wait')
  }
}
