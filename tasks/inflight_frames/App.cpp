#define STB_IMAGE_IMPLEMENTATION
#include "App.hpp"

App::App()
  : resolution{1280, 720}
  , useVsync{false}
{
  // Инициализация Vulkan с требуемыми расширениями
  {
    auto glfwInstExts = windowing.getRequiredVulkanInstanceExtensions();
    std::vector<const char*> instanceExtensions{glfwInstExts.begin(), glfwInstExts.end()};
    std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    etna::initialize(etna::InitParams{
      .applicationName = "Inflight Frames",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .instanceExtensions = instanceExtensions,
      .deviceExtensions = deviceExtensions,
      .physicalDeviceIndexOverride = {},
      .numFramesInFlight = FRAMES_IN_FLIGHT,
    });
  }

  // Создание окна ОС
  osWindow = windowing.createWindow(OsWindow::CreateInfo{
    .resolution = resolution,
  });

  // Привязка окна ОС к Vulkan
  {
    auto surface = osWindow->createVkSurface(etna::get_context().getInstance());
    vkWindow = etna::get_context().createWindow(etna::Window::CreateInfo{
      .surface = std::move(surface),
    });
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
    });
    // Обновляем разрешение согласно swapchain
    resolution = {w, h};
  }

  // Создаем менеджеры команд
  commandManager = etna::get_context().createPerFrameCmdMgr();
  oneShotManager = etna::get_context().createOneShotCmdMgr();

  // Создаем вычислительную программу и пайплайн
  etna::create_program("texture", {INFLIGHT_FRAMES_SHADERS_ROOT "texture.comp.spv"});
  computePipeline = etna::get_context().getPipelineManager().createComputePipeline("texture", {});
  sampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "computeSampler"});

  bufImage = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "output",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc |
                  vk::ImageUsageFlagBits::eSampled
  });

  // Создаем графическую программу и пайплайн
  etna::create_program(
    "image",
    {INFLIGHT_FRAMES_SHADERS_ROOT "toy.vert.spv", INFLIGHT_FRAMES_SHADERS_ROOT "toy.frag.spv"});
  graphicsPipeline = etna::get_context().getPipelineManager().createGraphicsPipeline(
    "image",
    etna::GraphicsPipeline::CreateInfo{
      .fragmentShaderOutput = {
        .colorAttachmentFormats = {vk::Format::eB8G8R8A8Srgb},
        .depthAttachmentFormat = vk::Format::eD32Sfloat,
      },
    });

  graphicsSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "graphicsSampler",
  });

  int width, height, channels;
  const auto file = stbi_load(
    INFLIGHT_FRAMES_SHADERS_ROOT "../../../../resources/textures/test_tex_1.png",
    &width,
    &height,
    &channels,
    STBI_rgb_alpha);

  // Проверка успешности загрузки текстуры
  if (!file) {
    throw std::runtime_error("Failed to load texture: test_tex_1.png");
  }

  image = etna::get_context().createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{static_cast<unsigned int>(width), static_cast<unsigned int>(height), 1},
    .name = "texture",
    .format = vk::Format::eR8G8B8A8Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eSampled
  });

  etna::BlockingTransferHelper(etna::BlockingTransferHelper::CreateInfo{
    .stagingSize = static_cast<std::uint32_t>(width * height),
  })
    .uploadImage(
      *oneShotManager,
      image,
      0,
      0,
      std::span(reinterpret_cast<const std::byte*>(file), width * height * 4));

  // Освобождаем память после загрузки текстуры
  stbi_image_free(file);


  // Создаем буферы для констант для каждого кадра
  for (auto& buffer : constantBuf) {
    buffer = etna::get_context().createBuffer(etna::Buffer::CreateInfo{
      .size = sizeof(params),
      .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU,
      .name = "const buffer"
    });
  }
}

App::~App()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::run()
{
  while (!osWindow->isBeingClosed())
  {
    windowing.poll();
    drawFrame();
  }
  // Ждем завершения всех команд перед закрытием приложения
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::drawFrame()
{
  // Получаем командный буфер для записи команд GPU
  auto currentCmdBuf = commandManager->acquireNext();
  etna::begin_frame();

  // Получаем изображение для рендера
  auto nextSwapchainImage = vkWindow->acquireNext();

  // Если окно свернуто, пропускаем кадры
  if (nextSwapchainImage)
  {
    auto [backbuffer, backbufferView, backbufferAvailableSem] = *nextSwapchainImage;

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));
    {
      ETNA_PROFILE_GPU(currentCmdBuf, "Render frame");

      std::this_thread::sleep_for(std::chrono::milliseconds(8));

      // Инициализируем backbuffer в нужное состояние для записи
      etna::set_state(
        currentCmdBuf,
        backbuffer,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageAspectFlagBits::eColor);

      auto computeShader = etna::get_shader_program("texture");
      auto computeDescriptorSet = etna::create_descriptor_set(
        computeShader.getDescriptorLayoutId(0),
        currentCmdBuf,
        {etna::Binding{0, bufImage.genBinding(sampler.get(), vk::ImageLayout::eGeneral)},
         etna::Binding{1, constantBuf[bufIndex].genBinding()}});
      const vk::DescriptorSet computeVkSet = computeDescriptorSet.getVkSet();

      currentCmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline.getVkPipeline());
      currentCmdBuf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        computePipeline.getVkPipelineLayout(),
        0,
        1,
        &computeVkSet,
        0,
        nullptr);

      int64_t elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now() - start)
                              .count();

      params = {
        .size_x = resolution.x,
        .size_y = resolution.y,
        .time = elapsedTime / 1000.f,
      };

      // Копируем параметры в константный буфер
      std::byte* data = constantBuf[bufIndex].map();
      std::memcpy(data, &params, sizeof(params));
      constantBuf[bufIndex].unmap();

      etna::flush_barriers(currentCmdBuf);

      auto imgInfo = etna::get_shader_program("image");
      auto graphicsSet = etna::create_descriptor_set(
        imgInfo.getDescriptorLayoutId(0),
        currentCmdBuf,
        {etna::Binding{0, bufImage.genBinding(sampler.get(), vk::ImageLayout::eGeneral)},
         etna::Binding{1, image.genBinding(graphicsSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
         etna::Binding{2, constantBuf[bufIndex].genBinding()}});
      const vk::DescriptorSet graphicsVkSet = graphicsSet.getVkSet();

      {
        ETNA_PROFILE_GPU(currentCmdBuf, "Render target");

        etna::RenderTargetState renderTargets{
          currentCmdBuf,
          {{0, 0}, {resolution.x, resolution.y}},
          {{.image = backbuffer, .view = backbufferView}},
          {}
        };

        currentCmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline.getVkPipeline());
        currentCmdBuf.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics,
          graphicsPipeline.getVkPipelineLayout(),
          0,
          1,
          &graphicsVkSet,
          0,
          nullptr);

        currentCmdBuf.draw(3, 1, 0, 0);
      }

      etna::flush_barriers(currentCmdBuf);


      // Обновляем индекс буфера для следующего кадра
      bufIndex = (bufIndex + 1) % FRAMES_IN_FLIGHT;

      // Переход backbuffer в состояние, пригодное для презентации
      etna::set_state(
        currentCmdBuf,
        backbuffer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        {},
        vk::ImageLayout::ePresentSrcKHR,
        vk::ImageAspectFlagBits::eColor);
      etna::flush_barriers(currentCmdBuf);

      ETNA_READ_BACK_GPU_PROFILING(currentCmdBuf);
    }
    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    auto renderingDone =
      commandManager->submit(std::move(currentCmdBuf), std::move(backbufferAvailableSem));

    const bool presented = vkWindow->present(std::move(renderingDone), backbufferView);
    if (!presented)
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  // Если окно не свернуто, но swapchain недоступен, пересоздаем его
  if (!nextSwapchainImage && osWindow->getResolution() != glm::uvec2{0, 0})
  {
    auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
      .resolution = {resolution.x, resolution.y},
      .vsync = useVsync,
    });
    ETNA_VERIFY((resolution == glm::uvec2{w, h}));
  }
}
